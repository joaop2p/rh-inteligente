# Instruções para agentes de IA — Sistema de RH Inteligente

Este arquivo define como agentes de IA devem atuar neste repositório. Leia também os documentos em `docs/` antes de propor alterações no modelo ou no pipeline.

## Regras inegociáveis
1. RAG, GraphRAG, KAG e Neo4j são requisitos do projeto; não remover nem substituir por uma busca vetorial simples.
2. Diferenciar claramente o que já está implementado, o que está especificado e o que é apenas proposta.
3. Não inventar fatos sobre o dataset. Consultar os arquivos saneados e a auditoria.
4. Não misturar empregado CLT, estagiário e prestador PJ.
5. Preservar IDs canônicos, direção das relações e temporalidade.
6. Não gerar respostas sem evidência documental/estrutural. Se não houver suporte, declarar insuficiência.
7. Não executar Cypher arbitrária gerada por LLM. Validar, parametrizar e restringir operações.
8. Não expor CPF, remuneração ou outros dados sensíveis desnecessariamente.
9. Não adicionar dependências, serviços ou abstrações sem justificativa objetiva.
10. Não afirmar que testes, ingestões ou comandos foram executados se não foram.

## Ordem de leitura
1. `README.md`
2. `docs/MODELAGEM_LPG_NOVATECH.md`
3. `docs/AUDITORIA_COMPLEMENTAR.md`
4. `neo4j/00_schema.cypher`
5. `requirements.txt`

## Forma de trabalho
- Trabalhar em incrementos pequenos e verificáveis.
- Antes de alterar o schema, explicar impacto e compatibilidade.
- Preferir funções pequenas, tipadas e testáveis.
- Separar ingestão, grafo, recuperação, geração e API.
- Tratar erros de conexão, documentos vazios, entidades não resolvidas e ausência de evidência.
- Escrever testes para regras temporais, vínculos e caminhos multi-hop.
- Atualizar README/documentação quando contratos ou decisões mudarem.
- Ao concluir uma tarefa, informar arquivos alterados, decisões tomadas, testes executados e pendências.

## Limites pedagógicos
O agente atua como mentor técnico. Deve explicar decisões, apresentar trechos de código e orientar depuração. Evitar entregar uma solução monolítica sem explicar os conceitos e sem permitir que a equipe compreenda a implementação.
