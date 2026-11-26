"""download.py — primeiro estagio: baixa as edicoes do DOM-SP para `raw/`."""
from __future__ import annotations

import datetime as dt
import io
import sys

from awsglue.utils import getResolvedOptions

from domsp.contrato import lago
from domsp.fonte import baixar

RAW = "raw"


def _datas(inicio: dt.date, fim: dt.date) -> list[dt.date]:
    """Todos os dias do intervalo, inclusive fim de semana e feriado."""
    if fim < inicio:
        raise ValueError(f"fim ({fim}) antes do inicio ({inicio})")
    return [inicio + dt.timedelta(days=n) for n in range((fim - inicio).days + 1)]


def main() -> int:
    argumentos = getResolvedOptions(sys.argv, ["bucket", "desde", "ate"])
    bucket = argumentos["bucket"]

    import boto3

    s3 = boto3.client("s3")

    desde, ate = lago.resolver_janela(argumentos["desde"], argumentos["ate"])
    # A coleta exige datas concretas em `desde`/`ate`: diferente do bronze/prata, nao ha como
    # "baixar tudo" sem saber entre quais datas.
    if desde == lago.SEM_LIMITE or ate == lago.SEM_LIMITE:
        print(
            f"janela da coleta sem limite ({desde!r}..{ate!r}): a coleta precisa de datas "
            f"concretas em `desde` e `ate` (AAAA-MM-DD) ou de `ultimos:N` em `desde`. "
            f"`{lago.SEM_LIMITE}` só vale para os estágios que leem o lago já existente.",
            file=sys.stderr,
        )
        return 2
    inicio = dt.date.fromisoformat(desde)
    fim = dt.date.fromisoformat(ate)

    baixadas = em_cache = sem_edicao = 0
    for data in _datas(inicio, fim):
        iso = data.isoformat()
        chave = f"{RAW}/dt={iso}/{baixar.nome_do_arquivo(iso)}"

        # Reexecutar nao rebaixa o que ja veio.
        try:
            s3.head_object(Bucket=bucket, Key=chave)
            em_cache += 1
            continue
        except s3.exceptions.ClientError:
            pass

        resposta = baixar.baixar(iso)
        if resposta.situacao == baixar.SEM_EDICAO:
            sem_edicao += 1
            continue
        if resposta.situacao == baixar.FALHA:
            raise RuntimeError(f"{iso}: {resposta.detalhe}")

        s3.upload_fileobj(io.BytesIO(resposta.conteudo), bucket, chave)
        baixadas += 1
        print(f"{iso}: {len(resposta.conteudo)} bytes")

    print(
        f"baixadas={baixadas} em_cache={em_cache} sem_edicao={sem_edicao} "
        f"intervalo={inicio}..{fim}"
    )

    if baixadas == 0 and em_cache == 0:
        print("ERRO: nenhuma edicao no periodo inteiro — a fonte provavelmente mudou")
        return 1
    return 0


if __name__ == "__main__":
    # Sem `raise SystemExit(main())`: o Glue executa com `runpy.run_path`, e o ProcessLauncher
    # trata QUALQUER SystemExit como erro — inclusive o de codigo 0.
    _codigo = main()
    if _codigo:
        raise SystemExit(_codigo)
