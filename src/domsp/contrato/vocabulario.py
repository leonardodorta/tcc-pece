"""
vocabulario.py — os conjuntos fechados de valores, declarados uma vez para os dois lados

## Por que este módulo existe, e por que ele é a exceção a uma regra do pacote

Por regra do pacote, `dominio` não importa `contrato`. Este módulo é a **única** exceção.

O vocabulário de valores fechados (os treze tipos de ato, as cinco naturezas, as nove
grandezas de valor, os seis motivos de descarte) é ao mesmo tempo:

  - **regra de domínio**, porque decidir que um apostilamento é `alteracao` e não
    `gasto_efetivo` é conhecimento sobre contratação pública; e
  - **termo do contrato**, porque é o conjunto de valores que quem consulta o Athena
    pode esperar na coluna, e está publicado no `Comment` que a interface exibe.

Por ser as duas coisas, o vocabulário tem uma declaração só, aqui, e a nota da coluna é
*composta* a partir dele (`esquema.Coluna.nota_completa`). Não há duas listas para manter em
sincronia, e portanto não há divergência possível entre o que a coluna promete e o que o
domínio produz. Na outra direção, se `dominio` produz algo que
não está aqui, ele erra alto na importação.

A exceção é estreita: `dominio` pode importar **este** módulo, e nenhum outro de `contrato`.
Este módulo, por sua vez, não importa nada — nem da biblioteca padrão. Ele é folha do grafo,
o que garante que a exceção não pode virar um caminho de volta para `lago` ou `esquema`.

## O que NÃO entra aqui

Os *mapeamentos* entre vocabulários. Que `ata_registro_preco` corresponda a
`reserva_preco` é decisão de domínio e mora em `dominio/ato.py`. Aqui ficam só os
**conjuntos**: quais valores existem.
"""
from __future__ import annotations

# Tipo do ato
#
# Os instrumentos que o art. 23 do Decreto 62.177/2023 — que altera o art. 2º do
# Decreto 46.195/2005 — manda publicar, restritos ao escopo deste projeto
# (contratação; licitação é descartada na triagem).
#
# A ordem é fixa, para que listas e notas geradas a partir daqui sejam determinísticas.

TIPOS: tuple[str, ...] = (
    "contrato",
    "aditivo",
    "doacao",
    "apostilamento",
    "ata_registro_preco",
    "penalidade",
    "empenho",
    "convenio",
    "termo_colaboracao",
    "termo_fomento",
    "acordo_cooperacao",
    "rescisao",
    "autorizacao_uso",
)

# Natureza — o que o ato faz com dinheiro público
#
# Este é o vocabulário que decide o que pode ser somado, e é a razão de ele existir separado
# do tipo. Somar ata de registro de preço junto com contrato executado, ou contar aditivo
# como despesa nova, produz agregado sem significado.

NATUREZAS: tuple[str, ...] = (
    "gasto_efetivo",     # despesa autorizada: contrato, empenho
    "reserva_preco",     # preço registrado e não executado: ata de registro de preço
    "repasse",           # transferência a terceiro setor: convênio, fomento, colaboração
    "alteracao",         # modifica ajuste existente: aditivo, apostilamento
    "nao_gasto",         # não é despesa: penalidade, rescisão, autorização de uso
)

# As naturezas cujos valores podem ser somados como despesa.
NATUREZAS_SOMAVEIS: frozenset[str] = frozenset({"gasto_efetivo"})

# Grandeza do valor — o que o número É
#
# Classificar errado a grandeza — tratar multa, valor mensal ou valor anual como valor
# contratual — é erro de classificação, não de localização do número. Publicar o valor sem a
# grandeza ao lado devolve ao consumidor a decisão que causa esse erro.

GRANDEZAS: tuple[str, ...] = (
    "total",         # o valor do ajuste
    "estimado",      # valor estimado, não fechado
    "anual",         # por ano
    "mensal",        # por mês
    "unitario",      # por unidade
    "parcela",       # uma parcela do total
    "penalidade",    # multa — NÃO é valor do ajuste
    "saldo",         # saldo remanescente
    "indefinido",    # há valor, e o texto não permite dizer de que grandeza
)

# As grandezas que representam o ajuste inteiro, e portanto as únicas somáveis.
# `indefinido` fica fora de propósito: somá-lo seria supor `total` sem evidência.
GRANDEZAS_SOMAVEIS: frozenset[str] = frozenset({"total", "estimado"})

# Triagem — por que uma publicação foi recusada, e em que estágio
#
# A triagem recusa publicações, e o motivo é publicado como dado, não como log:
# quem lê os resultados precisa ver que o descarte é filtro deliberado (licitação, ato de
# pessoal, série fora de escopo) e não perda silenciosa.

MOTIVOS_DESCARTE: tuple[str, ...] = (
    "caderno_fora_de_escopo",     # art. 13: Câmara, TCM
    "veiculo_fora_de_escopo",     # Servidores, Concursos
    "serie_de_licitacao",         # aviso, edital, chamamento — escopo declarado
    "serie_de_ruido",             # série que não veicula ato de contratação
    "sem_palavra_confirmadora",   # metadado em escopo, texto não confirma
    "sem_atos_publicados",        # art. 7º: edição sem atos oficiais
)

# Os dois estágios da triagem: as regras que decidem por metadado são aplicadas antes de
# converter HTML em texto.
ESTAGIO_METADADO = "metadado"
ESTAGIO_TEXTO = "texto"
ESTAGIOS: tuple[str, ...] = (ESTAGIO_METADADO, ESTAGIO_TEXTO)

# Resíduo — por que um ato sobrou para o modelo

MOTIVOS_RESIDUO: tuple[str, ...] = (
    "sem_campo_no_formulario",
    "sem_rotulo_na_prosa",
    "recusado_na_limpeza",
)

# Origem da chave de agrupamento
#
# A chave declara em que a igualdade se apoia, porque quem consulta precisa saber se um
# agrupamento vem de registro público (raiz de CNPJ) ou de semelhança de grafia.

ORIGEM_CNPJ = "cnpj"
ORIGEM_NOME = "nome"
ORIGENS_CHAVE: tuple[str, ...] = (ORIGEM_CNPJ, ORIGEM_NOME)

# Estrato da medição
#
# Métrica agregada sobre tarefas de dificuldade diferente sugere desempenho uniforme que não
# existe. Os estratos são o vocabulário que torna a média decomponível.

# O estrato que agrega todos os outros. Nomeado porque `pipeline/gold.py` o escreve em cada
# linha de `gold_quality` e consultas de qualidade o usam no `WHERE`: os lugares têm de
# concordar.
ESTRATO_AGREGADO = "agregado"

ESTRATOS: tuple[str, ...] = (ESTRATO_AGREGADO, "com_rotulo", "sem_rotulo", "residuo")

# Os métodos de intervalo de confiança em uso. Proporção publicada sem método de intervalo
# não é reproduzível: a aproximação normal devolveria [1; 1] para 60 acertos em 60, isto é,
# certeza a partir de sessenta observações.
# Wilson é o método usado nas proporções publicadas em `gold_quality`, nomeado pela mesma
# razão do estrato: o valor é gravado pelo pipeline na coluna `ci_method`, e a coluna declara
# `METODOS_INTERVALO` como domínio.
INTERVALO_WILSON = "wilson"

METODOS_INTERVALO: tuple[str, ...] = (INTERVALO_WILSON, "bootstrap", "katz")


# A guarda
#
# Valor repetido dentro de um vocabulário fechado é defeito: o conjunto e a tupla deixariam
# de ter o mesmo tamanho, e uma contagem por valor daria número diferente de uma contagem
# por posição.

for _nome, _vocab in (
    ("TIPOS", TIPOS), ("NATUREZAS", NATUREZAS), ("GRANDEZAS", GRANDEZAS),
    ("MOTIVOS_DESCARTE", MOTIVOS_DESCARTE), ("ESTAGIOS", ESTAGIOS),
    ("MOTIVOS_RESIDUO", MOTIVOS_RESIDUO), ("ORIGENS_CHAVE", ORIGENS_CHAVE),
    ("ESTRATOS", ESTRATOS), ("METODOS_INTERVALO", METODOS_INTERVALO),
):
    if len(set(_vocab)) != len(_vocab):
        raise AssertionError(f"{_nome} tem valor repetido")
    if any(v != v.lower() or " " in v for v in _vocab):
        raise AssertionError(
            f"{_nome} tem valor fora da convenção (minúscula, sem espaço): "
            f"{[v for v in _vocab if v != v.lower() or ' ' in v]}"
        )

if not set(GRANDEZAS) >= GRANDEZAS_SOMAVEIS:
    raise AssertionError("GRANDEZAS_SOMAVEIS tem valor fora de GRANDEZAS")
if not set(NATUREZAS) >= NATUREZAS_SOMAVEIS:
    raise AssertionError("NATUREZAS_SOMAVEIS tem valor fora de NATUREZAS")

del _nome, _vocab
