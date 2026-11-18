# Auditoria complementar da base saneada

Esta auditoria foi realizada sobre os CSVs e documentos TXT do pacote recebido. Ela não substitui nem reexecuta automaticamente as 62 validações mencionadas no relatório original.

## Correção aplicada

- `relacoes_funcionais.csv`, registro `RF-002`: direção invertida e nome da relação incompatível com a direção.
- Evidência: DOC-015, cláusula 1ª, identifica Ana Clara Monteiro (E013) como supervisora de Isabela Vasconcelos (E015), enquanto Camila Rocha Medeiros (E007) é a gestora hierárquica.
- Corrigido para `E013 -> E015`, `SUPERVISIONA_ESTAGIO`. DOC-037 reforça o papel de Camila na avaliação/prorrogação, sem mudar a distinção entre supervisão técnica e gestão hierárquica.

## Esclarecimento de contagem

- `cargo_mapeamento.csv`: 29 linhas.
- 28 denominações são cargos e mapeiam para 24 cargos canônicos.
- 1 linha (`Consultor de Segurança da Informação`) é uma descrição de serviço PJ (`NAO_MAPEADO_PJ`) e não deve gerar `Cargo`.
- `cargos.csv`: 24 registros.

## Conferências estruturais reproduzidas

- Pessoas: 27 (25 EMPREGADO, 1 ESTAGIARIO, 1 PRESTADOR_PJ).
- Departamentos: 9.
- Organizações: 3.
- Cargos canônicos: 24.
- Documentos indexados: 60.
- Relações hierárquicas: 29 (22 ATUAL, 4 encerradas por mudança de gestor, 3 encerradas por desligamento).
- Relações funcionais: 2, com a direção de RF-002 corrigida.

A versão original do dataset permanece preservada em separado; esta pasta é a versão corrigida para uso no projeto.
