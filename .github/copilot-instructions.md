# GitHub Copilot — instruções do repositório

Siga `AGENTS.md` como fonte principal de orientação.

- Use Python com tipagem e separação por responsabilidade.
- Use Neo4j Cypher parametrizado; nunca concatene entradas do usuário em consultas.
- Preserve o modelo LPG e as relações temporais documentadas.
- Não transforme prestador PJ em empregado nem supervisão técnica em subordinação hierárquica.
- Não trate similaridade vetorial isolada como GraphRAG.
- Respostas geradas devem carregar referências a `doc_id`, `chunk_id` e evidências do grafo.
- Não invente score de confiança.
- Não adicione dependências sem necessidade demonstrável.
- Crie ou atualize testes junto com mudanças de comportamento.
