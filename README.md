# Sistema de RH Inteligente — NovaTech

Backend de um sistema de Recursos Humanos que consulta dados organizacionais e documentos internos usando **RAG, GraphRAG e KAG**, com o Neo4j como base de conhecimento em modelo LPG (*Labeled Property Graph*).

> **Estado do projeto:** documentação inicial e modelagem LPG disponíveis. Esta documentação não afirma que os pipelines RAG/GraphRAG/KAG ou a API já estejam implementados.

## 1. Objetivo

Permitir que usuários façam perguntas sobre a estrutura organizacional e os documentos de RH da NovaTech, recebendo respostas fundamentadas em evidências recuperadas. O sistema deve expor a origem documental e o subgrafo utilizado, reduzindo respostas sem suporte.

## 2. Escopo funcional

### Escopo do backend
- Ingerir documentos TXT (e, futuramente, PDF/DOCX, se necessário).
- Extrair texto, dividir em chunks e gerar embeddings.
- Extrair entidades e relações (NER/RE) e resolver nomes para IDs canônicos.
- Persistir entidades, relacionamentos, documentos e chunks no Neo4j.
- Executar RAG vetorial, GraphRAG com Cypher e travessias multi-hop.
- Combinar evidências documentais e estruturais para o fluxo KAG.
- Retornar resposta, fontes, subgrafo, consulta Cypher e sinais observáveis de recuperação.

### Fora do primeiro incremento
- Interface web completa e autenticação/autorização de produção.
- Integrações com folha de pagamento, ponto, ERP ou sistemas externos.
- Decisões automáticas sobre contratação, demissão, promoção ou medidas disciplinares.
- Uso de dados reais de colaboradores.

A especificação acadêmica exige, na entrega final, interface funcional, visualização interativa do subgrafo e apresentação de rastreabilidade. O desenvolvimento atual pode começar pelo backend, mas a API deve preparar os dados necessários para essa interface.

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

## 7. Contrato sugerido da resposta da API

```json
{
  "answer": "Resposta fundamentada",
  "sources": [
    {
      "doc_id": "DOC-...",
      "chunk_id": "CHUNK-...",
      "quote": "Trecho que sustenta a resposta"
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
| Banco de grafos | Neo4j 5.26+ |
| Modelo de dados | LPG |
| Índices | Unicidade, range, full-text e vector |
| Embeddings | BAAI/bge-m3, dimensão 1024 (hipótese usada no schema; confirmar antes de ingerir) |
| Linguagem/API | Python + FastAPI (proposta inicial para o backend; confirmar no repositório) |
| LLM | Provedor/modelo ainda a definir |
| Orquestração | Implementação direta ou biblioteca; decidir conforme necessidade, sem adicionar dependências sem uso |

Antes da ingestão, confirmar versão do Neo4j, modelo de embeddings/dimensão e modo de execução (Docker, Desktop ou Aura). Se o modelo de embeddings mudar, ajustar o índice vetorial antes de popular os vetores.

## 9. Como começar

1. Clonar o repositório.
2. Criar e ativar ambiente virtual Python.
3. Instalar dependências: `pip install -r requirements.txt`.
4. Configurar variáveis a partir de `.env.example` (quando criado; nunca versionar segredos).
5. Iniciar Neo4j e configurar credenciais.
6. Executar `neo4j/00_schema.cypher` no Neo4j Browser.
7. Validar constraints e índices com `SHOW CONSTRAINTS;` e `SHOW INDEXES;`.
8. Implementar carga dos dados mestres antes de chunks/embeddings.
9. Executar testes de integridade e consultas de validação antes de conectar o LLM.

> Os passos 4–9 dependem de scripts/arquivos que ainda precisam ser implementados ou configurados. Não há comando de inicialização da API documentado enquanto a estrutura executável não existir.

### Importação de chunks e embeddings

Depois de importar os nós `Documento`, gere e valide os chunks com:

```bash
.venv/bin/python -m src.ingestion.import_chunks \
  --data-dir documents/downloads/novatech_proximos_passos_lpg/dataset_saneado_corrigido/dataset_saneado \
  --limit-documents 3
```

O comando é um *dry-run* para os três primeiros documentos do índice; remova `--limit-documents` para processar todos. CPF formatado e valores monetários precedidos por `R$` são mascarados antes de gerar embeddings, inclusive valores de benefícios. Essa sanitização não substitui uma revisão de privacidade. Para gravar `Chunk`, `embedding` e `TEM_TRECHO` no Neo4j, configure `NEO4J_URI` e `NEO4J_PASSWORD` no `.env` e acrescente `--apply`.

## 10. Estrutura sugerida do repositório

```text
.
├── README.md
├── requirements.txt
├── .env.example
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
│   ├── api/
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

- [ ] Schema Neo4j executa sem erros e constraints/índices são verificados.
- [ ] Nós e relações do dataset são carregados com IDs canônicos.
- [ ] Datas e vínculos históricos são preservados corretamente.
- [ ] Documentos são associados a chunks com referência ao arquivo de origem.
- [ ] Embeddings são gerados pelo mesmo modelo e dimensão configurados no índice.
- [ ] RAG recupera trechos relevantes e cita suas fontes.
- [ ] GraphRAG executa Cypher e demonstra ao menos uma travessia multi-hop.
- [ ] Recuperação híbrida combina e deduplica resultados vetoriais e do grafo.
- [ ] KAG fornece ao LLM fatos, relações, datas e evidências rastreáveis.
- [ ] API retorna fontes, nós, arestas, Cypher e métricas disponíveis.
- [ ] Quando não há evidência, o sistema declara a insuficiência em vez de fabricar resposta.
- [ ] Testes cobrem casos positivos, ausência de evidência e relações temporais.

## 13. Referências internas

- `docs/MODELAGEM_LPG_NOVATECH.md` — especificação detalhada do grafo.
- `docs/AUDITORIA_COMPLEMENTAR.md` — correção de relação funcional e conferências do dataset.
- `neo4j/00_schema.cypher` — constraints e índices iniciais.
- Especificação acadêmica do projeto — requisitos de Text-to-Graph, Neo4j LPG, recuperação híbrida/multi-hop e interface com explicabilidade.
