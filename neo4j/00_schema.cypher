// Schema mínimo usado pelos importadores e pela recuperação híbrida do projeto.
// Documento.escopo é criado pelo importador de metadados; a busca o filtra após
// localizar os chunks relacionados pelo relacionamento TEM_TRECHO.

CREATE CONSTRAINT pessoa_funcionario_id_unique IF NOT EXISTS
FOR (n:Pessoa) REQUIRE n.funcionario_id IS UNIQUE;

CREATE CONSTRAINT organizacao_organizacao_id_unique IF NOT EXISTS
FOR (n:Organizacao) REQUIRE n.organizacao_id IS UNIQUE;

CREATE CONSTRAINT departamento_departamento_id_unique IF NOT EXISTS
FOR (n:Departamento) REQUIRE n.departamento_id IS UNIQUE;

CREATE CONSTRAINT cargo_cargo_id_unique IF NOT EXISTS
FOR (n:Cargo) REQUIRE n.cargo_id IS UNIQUE;

CREATE CONSTRAINT documento_doc_id_unique IF NOT EXISTS
FOR (n:Documento) REQUIRE n.doc_id IS UNIQUE;

CREATE CONSTRAINT chunk_chunk_id_unique IF NOT EXISTS
FOR (n:Chunk) REQUIRE n.chunk_id IS UNIQUE;

CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1024,
  `vector.similarity_function`: 'cosine'
}};

CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
FOR (c:Chunk) ON EACH [c.texto];
