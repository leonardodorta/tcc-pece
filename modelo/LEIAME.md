# Modelo de reconhecimento da contratada

Pacote do SageMaker (`model.tar.gz`, 1,2 GB) com o BERTimbau Large ajustado a partir de `neuralmind/bert-large-portuguese-cased`.

Conteúdo: `model.safetensors`, `config.json`, tokenizador, `code/inference.py`, a wheel `domsp` 1.0.0 e os metadados do treino (`metadata.json`, `metadata_treino.json`).

O arquivo está dividido em partes de 90 MiB (`model.tar.gz.parte-00` a `-13`) porque o GitHub recusa arquivos acima de 100 MB. Para reconstituí-lo:

```bash
bash modelo/juntar.sh
```

O script concatena as partes e confere o SHA-256 (`19dde8aa…d30c8`). Depois, envie o `model.tar.gz` ao S3 e informe a URI em `ModelArtifactUri` (veja a seção "Modelo" do README).
