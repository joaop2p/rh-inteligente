# Modelagem LPG — Sistema de RH Inteligente (NovaTech)

## 1. Objetivo e limites

Este documento define a primeira versão do modelo **Labeled Property Graph (LPG)** no Neo4j para a base saneada da NovaTech. O modelo foi desenhado para sustentar:

- consultas estruturadas de RH e hierarquia;
- RAG documental por trechos (`Chunk`);
- GraphRAG com combinação de busca vetorial e travessias Cypher multi-hop;
- KAG, usando entidades, relações e evidências explícitas para restringir a geração;
- rastreabilidade: cada resposta deve apontar documentos/trechos e o subgrafo utilizado.

A implementação inicial cobre o dataset disponível: 27 pessoas, 24 cargos canônicos, 9 departamentos, 3 organizações e 60 documentos TXT. Não presume dados ausentes nem converte o prestador PJ E026 em empregado.

> O desenvolvimento atual pode permanecer backend-only. Porém, a especificação da disciplina também exige, na entrega final, interface funcional, visualização interativa do subgrafo, lista de nós consultados e consulta Cypher usada. A consulta deve vir de templates allowlisted, nunca de Cypher arbitrária do LLM. Exponha confiança somente se houver métrica calibrada; não invente score.

## 2. Entidades (nodes)

| Label | Chave única | Propriedades principais | Origem |
|---|---|---|---|
| `Pessoa` | `funcionario_id` | `nome`, `email`, `categoria_pessoa`, `tipo_vinculo`, `status`, `data_admissao`, `data_desligamento`, `departamento_id`, `cargo_atual_id`, `remuneracao_tipo` | `funcionarios.csv` |
| `Organizacao` | `organizacao_id` | `nome`, `tipo_organizacao`, `relacao_com_novatech`, `cnpj` | `organizacoes.csv` |
| `Departamento` | `departamento_id` | `nome` | `departamentos.csv` |
| `Cargo` | `cargo_id` | `cargo_canonico`, `funcao`, `senioridade`, `camada_gestao`, `tipo_posicao` | `cargos.csv` |
| `Documento` | `doc_id` | `titulo`, `tipo`, `categoria`, `data_documento`, `escopo`, `resumo`, `arquivo`, `tipo_vinculo_pessoa` | `documentos_index.csv` |
| `Chunk` | `chunk_id` | `texto`, `ordem`, `token_count`, `embedding`, `doc_id` | gerado no pipeline de ingestão |

### Decisões

1. `Pessoa` representa pessoas relacionadas à organização, não apenas empregados. `categoria_pessoa` separa `EMPREGADO`, `ESTAGIARIO` e `PRESTADOR_PJ`.
2. `Cargo` representa posição/função organizacional. Senioridade e camada de gestão são propriedades distintas. A descrição de serviço de E026 não gera nó `Cargo`.
3. `Documento` armazena metadados e ponteiro para o TXT original; o conteúdo dividido em partes fica em `Chunk`.
4. `Chunk.embedding` é a propriedade vetorial inicial. Não indexar CPF, remuneração ou conteúdo pessoal desnecessário em índices de busca textual.
5. Entidades extraídas automaticamente de documentos devem ser resolvidas contra IDs canônicos antes de criar nós/arestas. Não criar uma segunda pessoa apenas porque o nome aparece com grafia diferente.

## 3. Relacionamentos

| Padrão | Propriedades | Fonte / semântica |
|---|---|---|
| `(Pessoa)-[:ALOCADO_EM]->(Departamento)` | — | Departamento atual de `funcionarios.csv`; o dataset não fornece histórico de lotação. |
| `(Pessoa)-[:OCUPA_CARGO]->(Cargo)` | `historico_id`, `valid_from`, `valid_to`, `tipo_relacao`, `cargo_original_no_periodo`, `fonte_doc_ids` | `historico_cargos.csv`; manter períodos históricos. |
| `(Pessoa)-[:REPORTS_TO]->(Pessoa)` | `relacao_id`, `valid_from`, `valid_to`, `status`, `valid_from_fonte`, `valid_to_fonte`, `valid_to_regra` | `hierarquia.csv`; datas inclusivas conforme convenção do saneamento. |
| `(Pessoa)-[:EMPREGADO_DE]->(Organizacao)` | `vinculo_id`, `data_inicio`, `data_fim`, `doc_contrato_id` | Somente CLT. |
| `(Pessoa)-[:ESTAGIARIO_DE]->(Organizacao)` | `vinculo_id`, `data_inicio`, `data_fim`, `data_fim_prevista`, `doc_contrato_id` | Estágio; não classificar como CLT. |
| `(Pessoa)-[:REPRESENTANTE_DE_CONTRATADA]->(Organizacao)` | `vinculo_id`, `data_inicio`, `data_fim`, `doc_contrato_id` | Representante da empresa prestadora PJ. |
| `(Organizacao)-[:PRESTA_SERVICOS_PARA]->(Organizacao)` | `data_inicio`, `data_fim`, `vinculo_id`, `doc_contrato_id` | ORG-002 presta serviços para ORG-001. |
| `(Pessoa)-[:INTERFACE_TECNICA]->(Pessoa)` | `valid_from`, `valid_to`, `fonte_doc_ids` | E026 → E002; não significa subordinação. |
| `(Pessoa)-[:SUPERVISIONA_ESTAGIO]->(Pessoa)` | `valid_from`, `valid_to`, `fonte_doc_ids` | E013 → E015, conforme DOC-015. E007 permanece gestora hierárquica de E015. |
| `(Documento)-[:REFERE_SE_A]->(Pessoa)` | — | `funcionario_id` do índice. |
| `(Documento)-[:ASSINADO_POR]->(Pessoa)` | — | IDs de `signatarios_ids`. |
| `(Documento)-[:ENVOLVE]->(Pessoa)` | — | IDs de `outros_envolvidos_ids`. |
| `(Documento)-[:EMITIDO_POR]->(Organizacao)` | — | `organizacao_emissora_id`. |
| `(Documento)-[:ENVOLVE_ORGANIZACAO]->(Organizacao)` | — | `organizacoes_envolvidas_ids`. |
| `(Documento)-[:RELACIONADO_A]->(Documento)` | — | `docs_relacionados`. |
| `(Documento)-[:TEM_TRECHO]->(Chunk)` | `ordem` opcional | Criado no chunking. |
| `(Chunk)-[:MENCIONA]->(Pessoa\|Cargo\|Departamento\|Organizacao)` | `surface_form`, `start_char`, `end_char`, `extraction_method`, `review_status`; `confidence` não é preenchido sem calibração | Baseline determinístico registra a ocorrência literal no chunk; extração NER/RE semântica e revisão de aliases permanecem pendentes. |

**Atenção à direção:** `SUPERVISIONA_ESTAGIO` deve sair do supervisor e apontar para o estagiário. A relação hierárquica `REPORTS_TO` é outra relação, com significado diferente.

## 4. Temporalidade

O saneamento adotou intervalos inclusivos: uma relação vale em `d` quando:

```cypher
r.valid_from <= $data
AND (r.valid_to IS NULL OR r.valid_to >= $data)
```

Não usar apenas `status = 'ATUAL'` para consultas históricas. Para o organograma atual, pode-se filtrar `valid_to IS NULL` e, adicionalmente, validar `status = 'ATUAL'`.

O dataset não fornece histórico temporal de departamento. Não inferir mudanças de departamento a partir de promoções/cargos. `ALOCADO_EM` deve ser tratado como fotografia atual.

## 5. Constraints e índices

O arquivo versionado `neo4j/00_schema.cypher` cria:

- constraints de unicidade para as chaves dos seis labels;
- índice full-text `chunk_fulltext` sobre `Chunk.texto`;
- índice vetorial `chunk_embedding` sobre `Chunk.embedding`, com similaridade cosseno.

O índice vetorial usa dimensão **1024**, assumindo o modelo multilíngue `BAAI/bge-m3`. Se outro modelo de embeddings for selecionado, alterar `vector.dimensions` para a dimensão real antes de ingerir os vetores. Todos os vetores de um índice precisam ter a mesma dimensão e usar o mesmo modelo/versão. Índices range e índices full-text de outras propriedades permanecem fora deste schema mínimo; adicioná-los quando consultas do projeto efetivamente precisarem deles.

## 6. Ordem de carga

1. Executar `00_schema.cypher` no Neo4j.
2. Carregar `Organizacao`, `Departamento` e `Cargo`.
3. Carregar `Pessoa`.
4. Criar `ALOCADO_EM` e `OCUPA_CARGO`.
5. Criar `REPORTS_TO` e relações funcionais, preservando datas e direção.
6. Criar vínculos CLT/estágio/PJ e relação entre organizações.
7. Carregar metadados `Documento` e suas relações com pessoas/organizações/documentos.
8. Extrair texto, gerar `Chunk`, embeddings e `TEM_TRECHO`.
9. Executar NER/RE nos chunks, resolver entidades canônicas e criar `MENCIONA`/relações extraídas com proveniência.

No `LOAD CSV`, converter strings vazias em `null` antes de converter datas/números. Usar o diretório `import/` do Neo4j ou URI segura conforme o ambiente. Não alterar os TXT originais.

## 7. Consultas de validação e demonstração

### 7.1 Hierarquia atual

```cypher
MATCH (p:Pessoa)-[r:REPORTS_TO]->(g:Pessoa)
WHERE r.valid_to IS NULL AND r.status = 'ATUAL'
RETURN p.nome AS subordinado, g.nome AS gestor
ORDER BY gestor, subordinado;
```

### 7.2 Gestor de uma pessoa em uma data histórica

```cypher
MATCH (p:Pessoa {funcionario_id: $funcionario_id})-[r:REPORTS_TO]->(g:Pessoa)
WHERE r.valid_from <= date($data)
  AND (r.valid_to IS NULL OR r.valid_to >= date($data))
RETURN p.nome, g.nome, r.valid_from, r.valid_to, r.valid_from_fonte, r.valid_to_fonte;
```

### 7.3 CLT, estágio e PJ sem misturar categorias

```cypher
MATCH (p:Pessoa)-[r:EMPREGADO_DE|ESTAGIARIO_DE|REPRESENTANTE_DE_CONTRATADA]->(o:Organizacao)
RETURN p.nome, p.categoria_pessoa, p.tipo_vinculo, type(r) AS relacao, o.nome
ORDER BY p.categoria_pessoa, p.nome;
```

### 7.4 Multi-hop: subordinados diretos e indiretos de uma pessoa

```cypher
MATCH path = (raiz:Pessoa {funcionario_id: $gestor_id})<-[:REPORTS_TO*1..5]-(p:Pessoa)
WHERE all(r IN relationships(path) WHERE
  r.valid_from <= date($data) AND (r.valid_to IS NULL OR r.valid_to >= date($data)))
RETURN p.funcionario_id, p.nome, length(path) AS distancia
ORDER BY distancia, p.nome, p.funcionario_id;
```

### 7.5 Caminho de evidência de um documento

```cypher
MATCH (d:Documento {doc_id: $doc_id})-[:REFERE_SE_A|ASSINADO_POR|ENVOLVE]->(p:Pessoa)
RETURN d.doc_id, d.titulo, p.funcionario_id, p.nome;
```

## 8. Estratégia de recuperação híbrida (próxima etapa)

1. **RAG:** consultar `chunk_embedding` por similaridade e recuperar texto + metadados do documento.
2. **GraphRAG:** usar entidades identificadas na pergunta para consultar o grafo com Cypher; percorrer relações temporais e multi-hop; recuperar documentos/chunks conectados.
3. **Híbrida:** combinar os candidatos vetoriais e os candidatos do grafo, deduplicar por `chunk_id`/`doc_id`, reordenar e montar contexto.
4. **KAG:** construir um pacote de conhecimento com fatos, caminhos do grafo, datas de validade e trechos-fonte; o LLM só pode responder com base nesse pacote.
5. **Proveniência:** devolver IDs dos documentos/chunks, entidades, arestas, Cypher executada e scores. A confiança não deve ser inventada pelo LLM: deve ser calculada a partir de sinais observáveis (score vetorial, cobertura de evidências, consistência temporal e sucesso das validações).

Para cumprir a rubrica, selecionar pelo menos um algoritmo de grafo para enriquecer contexto (por exemplo, shortest path ou PageRank). O primeiro protótipo pode demonstrar caminho mínimo entre entidades; PageRank/Leiden podem ficar como extensão se houver tempo e justificativa.

## 9. Contrato mínimo esperado da API de resposta

```json
{
  "answer": "Resposta em linguagem natural",
  "sources": [{"doc_id": "DOC-...", "chunk_id": "...", "text": "texto completo do chunk recuperado"}],
  "graph": {
    "nodes": [{"id": "E...", "label": "Pessoa", "name": "..."}],
    "edges": [{"source": "E...", "target": "E...", "type": "REPORTS_TO"}]
  },
  "retrieval": {
    "mode": "HYBRID",
    "cypher": "...",
    "vector_scores": [],
    "confidence": null
  }
}
```

`confidence: null` é preferível a apresentar uma nota arbitrária antes de definir e calibrar a métrica.

## 10. Pendências antes da ingestão

- Confirmar versão do Neo4j (recomendado Neo4j 5.26+ para o conjunto de sintaxes do script).
- Confirmar o modelo de embeddings e sua dimensão; o script assume BGE-M3 / 1024.
- Decidir se a instalação será Docker ou Neo4j Desktop/Cloud e configurar o diretório de importação.
- Manter a pendência documental de E026 (lacuna entre os contratos) sem inventar continuidade.
- Não corrigir o DOC-060 no grafo como fato: ele é evidência de organograma, mas não prevalece sobre a ausência de subordinação contratual de E026.
