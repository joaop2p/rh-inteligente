# Sistema de RH Inteligente — NovaTech

Aplicação monolítica em Python e Streamlit para consultas de Recursos Humanos, com Neo4j como base de conhecimento em modelo LPG (*Labeled Property Graph*) e previsão de fluxos **RAG, GraphRAG e KAG**.

> **Estado do projeto:** os scripts de ingestão do cadastro mestre, relações, metadados documentais, chunks e embeddings foram executados no Neo4j. A recuperação híbrida e os pipelines RAG e GraphRAG com proveniência estão conectados à interface. O KAG foi habilitado como um pacote explícito de fatos, relações temporais, caminhos e chunks antes da geração. Conversas diretas, RAG e consultas KAG podem ser salvas localmente em arquivos Markdown criptografados. Conversa direta pode habilitar opcionalmente a consulta ao banco por KAG após consentimento.

## 1. Objetivo

Permitir que usuários façam perguntas sobre a estrutura organizacional e os documentos de RH da NovaTech, recebendo respostas fundamentadas em evidências recuperadas. O sistema deve expor a origem documental e o subgrafo utilizado, reduzindo respostas sem suporte.

## 2. Escopo funcional

### Escopo da aplicação
- Ingerir documentos TXT (e, futuramente, PDF/DOCX, se necessário).
- Extrair texto, dividir em chunks e gerar embeddings.
- Extrair menções exatas a nomes canônicos e ligá-las aos chunks; NER/RE semântico e resolução de aliases continuam pendentes.
- Persistir entidades, relacionamentos, documentos e chunks no Neo4j.
- Executar RAG vetorial e uma travessia GraphSearch hierárquica multi-hop com Cypher allowlisted.
- Combinar evidências documentais e estruturais para o fluxo KAG.
- Exibir resposta, fontes, subgrafo, consulta Cypher e sinais observáveis de recuperação.

### Fora do primeiro incremento
- Autenticação/autorização de produção.
- Integrações com folha de pagamento, ponto, ERP ou sistemas externos.
- Decisões automáticas sobre contratação, demissão, promoção ou medidas disciplinares.
- Uso de dados reais de colaboradores.

A especificação acadêmica exige, na entrega final, interface funcional, visualização interativa do subgrafo e apresentação de rastreabilidade. A aplicação Streamlit é o ponto de interação do sistema monolítico; busca e visualização de evidências ainda serão integradas.

## 3. Arquitetura conceitual

```text
Documentos e dataset saneado
          |
          v
  Pipeline de ingestão
  - leitura / extração
  - limpeza e chunking
  - embeddings
  - NER / RE
  - resolução de entidades
          |
          +-------------------------+
          |                         |
          v                         v
     RAG vetorial             Neo4j LPG
     (Chunk)              (pessoas, cargos,
          |                departamentos,
          |                organizações,
          |                documentos...)
          |                         |
          |                   Cypher / multi-hop
          |                         |
          +------------+------------+
                       v
              Recuperação híbrida
                       |
                       v
                Contexto KAG
        (fatos + caminhos + trechos-fonte)
                       |
                       v
                     LLM
                       |
                       v
        Resposta + fontes + subgrafo
        + Cypher + métricas observáveis
```

**Distinção obrigatória:**
- **RAG:** recuperação de chunks por similaridade vetorial e uso dos trechos documentais como contexto.
- **GraphRAG:** recuperação vetorial combinada com consulta Cypher, relações do grafo e travessia multi-hop.
- **KAG:** construção de contexto de conhecimento estruturado — fatos, relações, temporalidade e evidências — para restringir/fundamentar a geração.

Uma busca vetorial isolada não deve ser apresentada como GraphRAG.

## 4. Base de dados disponível

O pacote `novatech_proximos_passos_lpg.zip` contém a versão saneada/corrigida do dataset e os artefatos da modelagem.

| Elemento | Quantidade |
|---|---:|
| Pessoas | 27 |
| Empregados | 25 |
| Estagiários | 1 |
| Prestador PJ | 1 |
| Departamentos | 9 |
| Organizações | 3 |
| Cargos canônicos | 24 |
| Documentos indexados | 60 |
| Relações hierárquicas | 29 |
| Relações funcionais | 2 |

O prestador PJ deve permanecer distinguível de empregado CLT e estagiário. Não inferir vínculo empregatício nem criar cargo para uma descrição de serviço PJ.

**Atenção à divergência entre materiais:** uma definição preliminar anterior mencionava 15 funcionários, 3 departamentos, 9 cargos e 18 documentos. A modelagem LPG e a auditoria do dataset saneado mais recente registram 27 pessoas, 9 departamentos, 24 cargos canônicos e 60 documentos. Este README adota os números do pacote saneado atual; atualizar se uma nova versão validada do dataset for aprovada.

## 5. Modelo de conhecimento (Neo4j LPG)

### Nós
- `Pessoa` — `funcionario_id` como chave única.
- `Organizacao` — `organizacao_id`.
- `Departamento` — `departamento_id`.
- `Cargo` — `cargo_id`.
- `Documento` — `doc_id`.
- `Chunk` — `chunk_id`, texto, ordem, contagem de tokens e embedding.

### Relações principais
- `Pessoa-[:ALOCADO_EM]->Departamento`
- `Pessoa-[:OCUPA_CARGO]->Cargo` (com período/histórico)
- `Pessoa-[:REPORTS_TO]->Pessoa` (hierarquia temporal)
- `Pessoa-[:EMPREGADO_DE|ESTAGIARIO_DE]->Organizacao`
- `Pessoa-[:REPRESENTANTE_DE_CONTRATADA]->Organizacao`
- `Organizacao-[:PRESTA_SERVICOS_PARA]->Organizacao`
- `Pessoa-[:INTERFACE_TECNICA]->Pessoa`
- `Pessoa-[:SUPERVISIONA_ESTAGIO]->Pessoa`
- `Documento-[:REFERE_SE_A|ASSINADO_POR|ENVOLVE]->Pessoa`
- `Documento-[:EMITIDO_POR|ENVOLVE_ORGANIZACAO]->Organizacao`
- `Documento-[:RELACIONADO_A]->Documento`
- `Documento-[:TEM_TRECHO]->Chunk`
- `Chunk-[:MENCIONA]->Pessoa|Cargo|Departamento|Organizacao`

Consulte `docs/MODELAGEM_LPG_NOVATECH.md` e `neo4j/00_schema.cypher` para propriedades, semântica, temporalidade e consultas de validação.

### Regras importantes
1. Resolver entidades para IDs canônicos antes de criar relações.
2. Não confundir `REPORTS_TO`, `SUPERVISIONA_ESTAGIO` e `INTERFACE_TECNICA`.
3. Relações hierárquicas têm validade temporal; para consultas históricas, filtrar `valid_from` e `valid_to`.
4. O histórico de departamento não está disponível. `ALOCADO_EM` é uma fotografia atual.
5. Não inventar continuidade de vínculo documental ausente.
6. Não indexar CPF, remuneração ou outros dados pessoais desnecessários em busca textual.

## 6. Casos de uso de demonstração

### RAG
**Pergunta:** “Quais são as condições de férias previstas na política interna da NovaTech?”

Fluxo esperado: recuperar chunks relevantes da política, responder somente com o conteúdo suportado e citar documento/trecho.

### GraphRAG (multi-hop)
**Pergunta:** “Quais funcionários subordinados à gerente de Tecnologia possuem contratos temporários?”

Fluxo esperado:
1. Identificar a gerente e seu ID canônico.
2. Percorrer relações `REPORTS_TO` válidas na data consultada.
3. Identificar os vínculos dos subordinados.
4. Recuperar documentos/chunks relacionados aos vínculos.
5. Consolidar os resultados sem misturar CLT, estágio e PJ.
6. Exibir as evidências e o caminho do grafo.

A pergunta deve ser ajustada aos dados efetivamente presentes. Se o dataset não sustentar a premissa (por exemplo, não houver gerente de Tecnologia ou contrato temporário correspondente), o sistema deve informar que não encontrou evidência suficiente, em vez de inventar resultados.

## 7. Formato sugerido para resposta da aplicação

```json
{
  "answer": "Resposta fundamentada",
  "sources": [
    {
      "doc_id": "DOC-...",
      "chunk_id": "CHUNK-...",
      "text": "Texto completo do chunk recuperado e citado"
    }
  ],
  "graph": {
    "nodes": [],
    "edges": []
  },
  "retrieval": {
    "mode": "HYBRID",
    "cypher": "MATCH ...",
    "vector_scores": [],
    "confidence": null
  }
}
```

`confidence` deve permanecer `null` até que exista uma métrica definida e validada. Não permitir que o LLM invente uma nota de confiança. Retornar scores e sinais observáveis separados.

## 8. Stack e decisões pendentes

| Componente | Diretriz atual |
|---|---|
| Banco de grafos | Neo4j 5.26+ (procedures clássicas); `SEARCH` é usado quando suportado |
| Modelo de dados | LPG |
| Índices | Unicidade, full-text de chunks e vetor de embeddings; demais índices conforme demanda |
| Embeddings | BAAI/bge-m3, dimensão 1024 (usado e validado na ingestão executada) |
| Aplicação | Python + Streamlit, arquitetura monolítica |
| LLM | Ollama remoto via Tailscale, configurado por `TAILSCALE_MODEL_HOST` |
| Orquestração | Implementação direta ou biblioteca; decidir conforme necessidade, sem adicionar dependências sem uso |

O modelo BAAI/bge-m3 e a dimensão 1024 foram usados na geração dos embeddings. O índice vetorial `chunk_embedding` foi verificado no Neo4j como `ONLINE`, também com dimensão 1024. Se o modelo de embeddings mudar, ajustar o índice vetorial antes de popular novos vetores.

## 9. Configuração e estado atual

Para preparar uma instalação nova:

1. Clonar o repositório.
2. Criar e ativar ambiente virtual Python.
3. Instalar dependências: `pip install -r requirements.txt`.
4. Copiar `.env.exemple` para `.env` e configurar as variáveis locais; nunca versionar segredos. A variável de criptografia vazia no exemplo é intencional: para persistir conversas, gere uma chave Fernet e coloque o valor somente em `RH_CONVERSATION_ENCRYPTION_KEY` no `.env` local:

   ```bash
   .venv/bin/python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
   ```

   Gere a chave uma única vez e mantenha-a em segurança: sem ela, as conversas salvas não podem ser lidas.
5. Iniciar Neo4j e configurar credenciais.
6. Executar `neo4j/00_schema.cypher` no Neo4j Browser. O script cria as seis constraints de unicidade usadas pelos IDs canônicos, o índice vetorial `chunk_embedding` (`Chunk.embedding`, 1024 dimensões, cosseno) e o índice full-text `chunk_fulltext` (`Chunk.texto`), todos com `IF NOT EXISTS`. O importador de metadados grava `Documento.escopo`, usado como filtro pelas consultas de busca.
7. Validar constraints e índices com `SHOW CONSTRAINTS;` e `SHOW INDEXES;`.

Os documentos de modelagem e auditoria estão versionados em `docs/`. O dataset saneado permanece em `documents/`, que é ignorado pelo Git, e deve ser obtido separadamente para reproduzir a ingestão. No banco usado na ingestão descrita neste README, os dados mestres, relações, metadados documentais, chunks e embeddings já foram carregados. Os índices `chunk_embedding` e `chunk_fulltext` foram verificados como `ONLINE`; o schema versionado ainda não foi executado/verificado nesse banco. O modo RAG está conectado ao backend e à interface; uma validação manual do fluxo real no navegador ainda é recomendada.

### Executar a interface

Com o Ollama acessível via Tailscale e `TAILSCALE_MODEL_HOST` configurado no `.env`, execute na raiz do projeto:

```bash
.venv/bin/streamlit run app.py
```

O chat mantém conversas diretas e RAG separadas, oferece retomada/exclusão pelo painel e temas claro/escuro. As conversas são gravadas como Markdown UTF-8 criptografado com Fernet em `.tmp/conversations/`, com UUID no nome e permissões locais restritas. A interface expurga conversas cuja última atividade passou de 30 dias sempre que executa; se o aplicativo estiver fechado, o expurgo ocorre na próxima execução. A chave fica em `RH_CONVERSATION_ENCRYPTION_KEY` no `.env`, que não deve ser compartilhado nem versionado. A criptografia protege os arquivos sem a chave, mas não contra alguém com acesso à conta local e ao `.env`. Se a chave não estiver configurada, o aplicativo informa que a persistência está indisponível e mantém o chat apenas na sessão atual.

Em **Conversa direta**, as mensagens são enviadas ao Ollama sem consultar o banco por padrão; a opção explícita "Consultar o banco Neo4j nesta conversa" ativa o fluxo KAG após consentimento. **RAG** consulta os índices Neo4j e mostra as citações documentais. **GraphRAG** atende perguntas hierárquicas suportadas e combina caminho temporal com chunks. **KAG** monta um pacote rastreável com fatos, relações temporais, caminhos e chunks antes da geração. RAG, GraphRAG e KAG enviam dados externos ao Ollama via Tailscale somente após consentimento explícito. Use dados saneados e confirme autorização, especialmente no escopo `PESSOA`. O histórico GraphRAG/KAG não é persistido separadamente; visualização interativa do subgrafo continua pendente.

O acesso ao Neo4j está encapsulado em `src/database/neo4j.py`, com transações de leitura/escrita e parâmetros Cypher separados da consulta. A geração em `src/generation/llm.py` usa a API nativa de chat do Ollama no host indicado por `TAILSCALE_MODEL_HOST` (URL ou host/porta, normalmente com a porta padrão do Ollama). Se houver exatamente um modelo instalado, ele é selecionado automaticamente. Com mais de um modelo, informe `model_name` ao criar `OllamaClient`.

### Baseline Text-to-Graph

`src/graph/text_to_graph.py` compara os nomes canônicos existentes em `Pessoa`, `Organizacao`, `Departamento` e `Cargo` com os textos de `Chunk`. Ao encontrar um nome completo (ignorando apenas maiúsculas/minúsculas e permitindo espaços/quebras de linha entre partes), produz uma relação `Chunk-[:MENCIONA]->Entidade` com o trecho literal e os offsets no chunk. Nomes iguais associados a mais de um ID ou label são tratados como ambíguos e não geram relação.

Execute primeiro em modo de simulação:

```bash
.venv/bin/python -m src.graph.text_to_graph
```

Para persistir as relações determinísticas:

```bash
.venv/bin/python -m src.graph.text_to_graph --apply
```

O modo padrão não altera o banco. `--apply` cria ou atualiza somente relações `MENCIONA`, usando queries Cypher fixas e parâmetros; não cria entidades, aliases ou relações organizacionais. O baseline não usa fuzzy matching, não infere relações semânticas e não gera score de confiança. Portanto, não equivale à extração geral NER/RE descrita no Projeto Final; aliases e entidades não catalogadas seguem sem resolução.

### GraphSearch hierárquico multi-hop

`src/retrieval/graph_search.py` oferece uma consulta backend para percorrer `REPORTS_TO` no sentido subordinado → gestor, com limite fixo de cinco saltos. A validade usa os intervalos inclusivos documentados (`valid_from <= as_of <= valid_to`, ou sem fim quando `valid_to` é nulo). A resolução por nome é exata, sem fuzzy matching; se o nome for ambíguo, a busca exige o ID canônico. O cargo CEO também pode ser resolvido somente por correspondência exata com `Cargo.cargo_canonico` igual a `CEO` ou `Chief Executive Officer`, considerando as validades de `OCUPA_CARGO` e do vínculo `EMPREGADO_DE` com a organização canônica `ORG-001`. A consulta é uma constante Cypher parametrizada, nunca texto produzido pelo LLM.

```python
from datetime import date

from src.database.neo4j import Neo4jDatabase
from src.retrieval.graph_search import GraphSearch

with Neo4jDatabase.from_env() as graph:
    search = GraphSearch(graph)
    manager = search.resolve_person_name("Nome canônico do gestor")
    if manager is not None:
        result = search.reporting_chain(
            manager.funcionario_id,
            as_of=date(2025, 1, 1),
        )
        print(result.paths)
        print(result.evidence)
```

O modo GraphRAG da interface combina esse caminho com resultados vetoriais/full-text e chunks `MENCIONA`, deduplicados por `doc_id`/`chunk_id`. A busca de chunks no grafo e nos índices respeita o mesmo escopo documental escolhido (`Documento.escopo`). A resposta cita somente chunks recuperados; o caminho, IDs e direção das relações temporais e referências de origem ficam expostos ao lado da resposta. O modo aceita perguntas hierárquicas contendo exatamente um nome canônico completo ou o cargo CEO com uma atribuição temporal única para `CEO`/`Chief Executive Officer`, vinculada à organização `ORG-001` por `EMPREGADO_DE`; cargos ausentes ou ambíguos resultam em insuficiência sem chamada ao LLM. Para pedidos pelos três subordinados mais próximos, retorna no máximo três pessoas com menor distância em `REPORTS_TO`, desempata pelo nome sem distinção entre maiúsculas/minúsculas e pelo ID canônico. A referência temporal usa a data atual, sem seletor na interface. A consulta é Cypher fixa/parametrizada, nunca gerada pelo modelo.

Esse fluxo é limitado ao caso hierárquico suportado por `REPORTS_TO`, com consultas Cypher parametrizadas e allowlisted. A conversa GraphRAG/KAG permanece apenas na sessão atual, sem persistência no armazenamento criptografado. O pacote KAG expõe fatos, relações temporais, caminho, `doc_id`/`chunk_id` e consulta usada. PageRank/Leiden, visualização interativa do subgrafo, aliases e NER/RE semântico permanecem pendentes.

### Recuperação híbrida inicial e pipeline RAG

`src/retrieval/vector_search.py` consulta os índices Neo4j `chunk_embedding` e `chunk_fulltext` e combina as listas por Reciprocal Rank Fusion (RRF). Não compara diretamente score vetorial e score lexical. `RetrievedChunk.score` é a similaridade vetorial (ou `None` se o resultado veio somente da busca lexical); `fulltext_score` é o score lexical bruto; `rank_score` é o sinal usado para ordenar a fusão. Nenhum desses valores representa confiança.

`VectorSearch.search` aceita `document_scope="ORGANIZACAO"` ou `"PESSOA"` para restringir explicitamente a busca ao escopo dos metadados documentais. O escopo não é inferido da pergunta. `min_vector_score` é um filtro opcional aplicado às pontuações vetoriais depois da fusão e antes de limitar a lista a `top_k`. Quando configurado, chunks com `score` abaixo do limiar são removidos; candidatos exclusivamente lexicais (`score is None`) são mantidos, pois não têm score vetorial comparável. A constante `RAG_MIN_VECTOR_SCORE`, em `src/rag/pipeline.py`, controla o valor padrão do pipeline e permanece `None` até uma calibração com a integração real. Portanto, por padrão a busca mantém o comportamento anterior, sem filtro.

Para calibrar, execute os testes de integração com Neo4j e BGE-M3 reais, registre `RetrievedChunk.score` dos chunks-alvo nas dez perguntas revisadas e compare-os com resultados de perguntas fora do domínio. Escolha um limiar somente após avaliar quantos alvos relevantes ele manteria e quantos resultados irrelevantes excluiria; documente os falsos positivos e negativos. Se a integração real não estiver disponível, mantenha `RAG_MIN_VECTOR_SCORE = None` em vez de assumir um valor. Similaridade vetorial, score full-text e RRF são sinais de recuperação, **não scores de confiança**.

`answer_with_rag` tenta primeiro `SearchClauseVectorSearch`, que usa a cláusula Cypher 25 `SEARCH`. Se o Neo4j retornar `CypherSyntaxError`, o app registra o fallback e repete a busca com `VectorSearch`, que usa as procedures clássicas `db.index.vector.queryNodes` e `db.index.fulltext.queryNodes`. Outros erros do Neo4j não são tratados como falta de suporte e continuam visíveis. O caminho de procedures é a compatibilidade mínima documentada para Neo4j 5.26+; `SEARCH` é uma otimização opcional. A documentação oficial declara suporte vetorial por `SEARCH` a partir do Neo4j 2026.01 e full-text a partir do 2026.09. Embora tenha funcionado na instância Aura 5.27 verificada, isso não garante disponibilidade em outras instâncias.

Exemplo de uso:

```python
from src.database.neo4j import Neo4jDatabase
from src.retrieval.vector_search import SearchClauseVectorSearch, load_embedding_model

with Neo4jDatabase.from_env() as graph:
    chunks = SearchClauseVectorSearch(graph, load_embedding_model()).search(
        "Quais são os critérios de promoção?",
        top_k=5,
        document_scope="ORGANIZACAO",
    )
    for chunk in chunks:
        print(
            chunk.doc_id,
            chunk.chunk_id,
            chunk.score,
            chunk.fulltext_score,
            chunk.rank_score,
            chunk.texto,
        )
```

`src/rag/pipeline.py` usa os resultados como candidatos documentais e solicita ao Ollama JSON com status, resposta e citações. Se o filtro configurado não deixar chunks, o pipeline retorna insuficiência sem chamar o LLM. A instrução restringe o modelo a RH da NovaTech e às evidências daquela pergunta; o histórico de conversa não é enviado como prova. Respostas citadas são aceitas somente se os pares `doc_id`/`chunk_id` existirem entre os trechos recuperados. `RAGSource.text` é o texto completo do chunk citado, não uma frase extraída ou validada como citação literal. Sem resultados, o LLM não é chamado; se o modelo marcar insuficiência, o pipeline retorna a mensagem fixa sem fontes.

Isso não garante, por si só, a validade semântica da resposta: o modelo ainda pode classificar incorretamente um trecho como suficiente. A qualidade da relevância e a insuficiência semântica ainda precisam de avaliação mais abrangente. O fluxo GraphRAG/KAG atual cobre somente a hierarquia por `REPORTS_TO`; outras consultas estruturadas, algoritmos de grafo e visualização interativa continuam pendentes.

Para testar a busca contra o Neo4j configurado e o modelo BGE-M3 real, com os dados saneados já carregados:

```bash
RH_RUN_NEO4J_INTEGRATION=1 .venv/bin/python -m pytest -q \
  tests/test_vector_search_integration.py \
  tests/test_rag_pipeline_integration.py
```

Os testes incluem dez perguntas com trechos-alvo revisados e um fluxo do pipeline com recuperação real e LLM simulado. Os testes automatizados não chamam o Ollama e não substituem uma avaliação abrangente. Em uma verificação manual no terminal com `qwen3.5:9b`, a pergunta sobre critérios de promoção recebeu resposta com citação válida usando somente a frase relevante da política. Em uma pergunta sobre quem foi promovido nos últimos 12 meses, o modelo retornou `insufficient` quando recebeu somente a evidência que indicava uma possível promoção, sem confirmação. A busca pode retornar candidatos para perguntas fora do domínio ou sem suporte; scores não são evidência suficiente por si sós.

Na instância verificada, `dbms.components()` informou Neo4j `5.27-aura` e Cypher 5/25; as consultas vetorial e full-text com `SEARCH` foram executadas sem os avisos de depreciação. Essa disponibilidade observada é específica à instância: a documentação oficial indica suporte geral a vetores por `SEARCH` a partir do Neo4j 2026.01 e full-text a partir do 2026.09. A rota de compatibilidade continua disponível em `VectorSearch`; valide suporte na sua própria instância antes de selecionar `SearchClauseVectorSearch`. Consulte a [documentação oficial da cláusula SEARCH](https://neo4j.com/docs/cypher-manual/current/clauses/search/) e a [visão geral dos índices](https://neo4j.com/docs/cypher-manual/current/indexes/search-indexes/).

Exemplo de uso:

```python
from src.database.neo4j import Neo4jDatabase
from src.generation.llm import OllamaClient

with Neo4jDatabase.from_env() as graph:
    graph.verify_connectivity()
    people = graph.execute_read(
        "MATCH (p:Pessoa {funcionario_id: $id}) RETURN p.nome AS nome",
        {"id": "E001"},
    )

answer = OllamaClient.from_env().chat([
    {"role": "user", "content": "Responda apenas: conexão validada."}
])
```

O conteúdo enviado a `OllamaClient.chat` é transmitido ao servidor configurado em Tailscale. Não envie documentos ou dados pessoais sem autorização e avaliação de privacidade.

Para conversar com o modelo em modo de teste, execute na raiz do projeto:

```bash
.venv/bin/python -m src.generation.chat_cli
```

Digite `/sair` para encerrar. A conversa mantém o histórico em memória durante a sessão, mas **não consulta o Neo4j nem usa RAG**; respostas sobre o dataset não são fundamentadas. Se o servidor tiver mais de um modelo, acrescente `--model NOME_DO_MODELO`. Não envie dados sensíveis.

Para executar os testes unitários:

```bash
.venv/bin/pytest -q
```

### Estado da ingestão e validação

No banco Neo4j configurado foram conferidas estas contagens após as importações:

| Tipo | Quantidade |
|---|---:|
| Organizacao | 3 |
| Departamento | 9 |
| Cargo | 24 |
| Pessoa | 27 |
| Relações cadastrais e vínculos | 115 |
| Documento | 60 |
| Relações documentais | 312 |
| Chunk | 115 |
| Relações `TEM_TRECHO` | 115 |

As relações cadastrais incluem 25 `EMPREGADO_DE`, 1 `ESTAGIARIO_DE` e 1 `REPRESENTANTE_DE_CONTRATADA`. `PRESTA_SERVICOS_PARA` não foi criada devido à lacuna documental sobre a continuidade do contrato PJ. Os chunks foram gerados com BAAI/bge-m3; foram validados com 1024 dimensões e valores finitos/normalizados. Os testes focados de chunking e embeddings passaram (17 testes).

Para repetir somente uma validação local de três documentos, sem gravar no banco:

```bash
.venv/bin/python -m src.ingestion.import_chunks \
  --data-dir documents/downloads/novatech_proximos_passos_lpg/dataset_saneado_corrigido/dataset_saneado \
  --limit-documents 3
```

O comando é um *dry-run* para os três primeiros documentos do índice; remova `--limit-documents` para processar todos. CPF formatado e valores monetários precedidos por `R$` são mascarados antes de gerar embeddings, inclusive valores de benefícios. Essa sanitização não substitui uma revisão de privacidade. A importação completa já foi aplicada no banco configurado; use `--apply` somente se quiser gravar novamente.

## 10. Estrutura sugerida do repositório

```text
.
├── app.py
├── README.md
├── requirements.txt
├── .env.exemple
├── .gitignore
├── AGENTS.md
├── .github/
│   └── agents/
├── docs/
│   ├── MODELAGEM_LPG_NOVATECH.md
│   └── AUDITORIA_COMPLEMENTAR.md
├── neo4j/
│   └── 00_schema.cypher
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── ingestion/
│   ├── graph/
│   ├── retrieval/
│   ├── generation/
│   └── core/
└── tests/
```

Esta é uma estrutura-alvo, não uma afirmação de que todos os diretórios já existem.

## 11. Segurança, privacidade e limites

- Usar somente dataset fictício/saneado para demonstração.
- Não enviar documentos ou dados pessoais a provedores externos sem autorização e avaliação de privacidade.
- Manter segredos em variáveis de ambiente; não commitar `.env`.
- Validar e parametrizar consultas Cypher; não executar texto arbitrário gerado pelo LLM sem validação.
- Aplicar controle de acesso antes de expor documentos, especialmente contratos e registros disciplinares.
- Respostas são apoio à consulta documental, não aconselhamento jurídico nem decisão automatizada de RH.
- Registrar fontes e proveniência para permitir auditoria.

## 12. Critérios mínimos de aceite

- [ ] Schema versionado executado sem erros; constraints e índices validados no Neo4j.
- [x] Nós e relações do dataset são carregados com IDs canônicos.
- [x] Baseline Text-to-Graph cria relações `MENCIONA` somente para correspondências exatas e não ambíguas com nomes canônicos.
- [x] Datas e propriedades temporais dos vínculos foram validadas e importadas.
- [x] Documentos são associados a chunks com referência ao arquivo de origem.
- [x] Embeddings são gerados pelo mesmo modelo e dimensão configurados no índice.
- [ ] RAG recupera trechos relevantes e cita suas fontes.
- [ ] GraphRAG executa Cypher e demonstra ao menos uma travessia multi-hop.
- [ ] Recuperação híbrida combina e deduplica resultados vetoriais e do grafo.
- [x] KAG fornece ao LLM fatos, relações, datas e evidências rastreáveis.
- [ ] Aplicação exibe fontes, nós, arestas, Cypher e métricas disponíveis.
- [ ] Quando não há evidência, o sistema declara a insuficiência em vez de fabricar resposta.
- [ ] Testes cobrem casos positivos, ausência de evidência e relações temporais.

Os testes unitários de `Neo4jDatabase` e `OllamaClient` usam drivers/respostas simulados; uma verificação de integração deve ser feita separadamente com o Neo4j e o servidor Ollama configurados.

## 13. Referências internas

- `docs/MODELAGEM_LPG_NOVATECH.md` — especificação detalhada do grafo.
- `docs/AUDITORIA_COMPLEMENTAR.md` — correção de relação funcional e conferências do dataset.
- `neo4j/00_schema.cypher` — constraints e índices iniciais.
- Especificação acadêmica do projeto — requisitos de Text-to-Graph, Neo4j LPG, recuperação híbrida/multi-hop e interface com explicabilidade.
