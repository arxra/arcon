use crate::data::arrow::ToArrow;
use arrow::{
    array::Array,
    chunk::Chunk,
    datatypes::Schema,
    error::ArrowError,
    io::{
        flight::{deserialize_batch, serialize_batch},
        ipc::{
            self,
            read::{deserialize_schema, read_file_metadata},
            write::{default_ipc_fields, schema_to_bytes},
        },
        parquet::{self, read::RecordReader},
    },
};
use arrow_format::flight::data::FlightData;
use std::path::Path;
use std::{collections::HashMap, fs::File};
use std::{convert::TryFrom, sync::Arc};

// Size for each RecordBatch in Arrow
pub const RECORD_BATCH_SIZE: usize = 1024;

// Looks the same as the old type
#[derive(Debug, Clone)]
pub struct SchemedChunk {
    chunk: Chunk<Arc<dyn Array>>,
    schema: Arc<Schema>,
}

impl SchemedChunk {
    pub fn new(chunk: Chunk<Arc<dyn Array>>, schema: Arc<Schema>) -> Self {
        Self { chunk, schema }
    }

    fn len(&self) -> usize {
        self.chunk.len()
    }
}

/// A Mutable Append Only Table
#[derive(Debug)]
pub struct MutableTable {
    /// Builder used to append data to the table
    builder: RecordBatchBuilder,
    /// Stores appended record batches
    batches: Vec<SchemedChunk>,
}
impl MutableTable {
    /// Creates a new MutableTable
    pub fn new(builder: RecordBatchBuilder) -> Self {
        Self {
            builder,
            batches: Vec::new(),
        }
    }

    /// Append a single element to the table
    #[inline]
    pub fn append(&mut self, elem: impl ToArrow, timestamp: Option<u64>) -> Result<(), ArrowError> {
        if self.builder.len() == RECORD_BATCH_SIZE {
            let batch = self.builder.record_batch();
            self.batches.push(batch);
        }

        self.builder.append(elem, timestamp)?;
        Ok(())
    }
    /// Load elements into the table from an Iterator
    #[inline]
    pub fn load(
        &mut self,
        elems: impl IntoIterator<Item = impl ToArrow>,
    ) -> Result<(), ArrowError> {
        for elem in elems {
            self.append(elem, None)?;
        }
        Ok(())
    }

    // internal helper to finish last batch
    fn finish(&mut self) -> Result<(), ArrowError> {
        if !self.builder.is_empty() {
            let batch = self.builder.record_batch();
            self.batches.push(batch);
        }
        Ok(())
    }

    /// Converts the MutableTable into an ImmutableTable
    pub fn immutable(mut self) -> Result<ImmutableTable, ArrowError> {
        self.finish()?;

        Ok(ImmutableTable {
            name: self.builder.name().to_string(),
            schema: self.builder.schema(),
            batches: self.batches,
        })
    }

    #[inline]
    pub fn batches(&mut self) -> Result<Vec<SchemedChunk>, ArrowError> {
        self.finish()?;
        let mut batches = Vec::new();
        std::mem::swap(&mut batches, &mut self.batches);
        Ok(batches)
    }

    #[inline]
    pub fn raw_batches(&mut self) -> Result<Vec<RawChunk>, ArrowError> {
        self.finish()?;
        let batches = self.batches()?;
        to_raw_batches(batches)
    }
}

#[derive(Debug)]
pub struct RecordBatchBuilder {
    table_name: String,
    schema: Arc<Schema>,
    builder: Vec<Arc<dyn Array>>,
}

impl RecordBatchBuilder {
    pub fn new(table_name: String, schema: Schema, builder: Vec<Arc<dyn Array>>) -> Self {
        Self {
            table_name,
            schema: Arc::new(schema),
            builder,
        }
    }
    #[inline]
    pub fn append(&mut self, elem: impl ToArrow, timestamp: Option<u64>) -> Result<(), ArrowError> {
        elem.append(&mut self.builder, timestamp)
    }
    pub fn name(&self) -> &str {
        &self.table_name
    }
    pub fn len(&self) -> usize {
        self.builder.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }
    pub fn record_batch(&mut self) -> SchemedChunk {
        SchemedChunk::new(Chunk::new(self.builder.clone()), self.schema())
    }
    pub fn set_name(&mut self, name: &str) {
        self.table_name = name.to_string();
    }
}

/// An Immutable Table
#[derive(Clone)]
pub struct ImmutableTable {
    pub(crate) name: String,
    pub(crate) schema: Arc<Schema>,
    pub(crate) batches: Vec<SchemedChunk>,
}

impl ImmutableTable {
    pub fn name(&self) -> String {
        self.name.clone()
    }
    pub fn total_rows(&self) -> usize {
        self.batches.iter().map(|r| r.len()).sum()
    }
    pub fn schema(&self) -> Arc<Schema> {
        self.schema.clone()
    }
    pub fn set_name(&mut self, name: &str) {
        self.name = name.to_string();
    }
}

#[inline]
pub fn to_record_batches(raw_batches: Vec<RawChunk>) -> Result<Vec<SchemedChunk>, ArrowError> {
    let mut chunks: Vec<SchemedChunk> = Vec::with_capacity(raw_batches.len());

    for raw in raw_batches {
        let (schema, ipc_schema) = deserialize_schema(&raw.schema)?;
        let fields = &schema.fields;
        let dictionaries = HashMap::new();
        let chunk = deserialize_batch(&raw.arrow_data, fields, &ipc_schema, &dictionaries)?;
        let schema = Arc::new(schema);

        chunks.push(SchemedChunk { chunk, schema });
    }
    Ok(chunks)
}

#[inline]
pub fn to_raw_batches(batches: Vec<SchemedChunk>) -> Result<Vec<RawChunk>, ArrowError> {
    let mut raw_batches = Vec::with_capacity(batches.len());

    for batch in batches {
        let ipc_fields = default_ipc_fields(&batch.schema.fields);
        let schema = schema_to_bytes(&batch.schema, &ipc_fields);
        let (dictionaries, arrow_data) = serialize_batch(
            &batch.chunk,
            &ipc_fields,
            &ipc::write::WriteOptions { compression: None },
        );

        raw_batches.push(RawChunk {
            schema,
            dictionaries,
            arrow_data,
        });
    }

    Ok(raw_batches)
}

/// Restore a ImmutableTable from a RawTable
impl TryFrom<RawTable> for ImmutableTable {
    type Error = ArrowError;

    fn try_from(table: RawTable) -> Result<Self, Self::Error> {
        let (s, _) = deserialize_schema(&table.schema)?;
        let schema = Arc::new(s);
        let batches = to_record_batches(table.batches)?;

        Ok(ImmutableTable {
            name: table.name,
            schema,
            batches,
        })
    }
}

/// A Raw version of an Arrow RecordBatch
#[derive(prost::Message, Clone)]
pub struct RawChunk {
    #[prost(bytes)]
    pub schema: Vec<u8>,
    #[prost(message, repeated)]
    pub dictionaries: Vec<FlightData>,
    #[prost(message, required)]
    pub arrow_data: FlightData,
}

/// A Raw version of [ImmutableTable] that can be persisted to disk or sent over the wire.
#[derive(prost::Message, Clone)]
pub struct RawTable {
    #[prost(string)]
    pub name: String,
    #[prost(bytes)]
    pub schema: Vec<u8>,
    #[prost(message, repeated)]
    pub batches: Vec<RawChunk>,
}

impl TryFrom<ImmutableTable> for RawTable {
    type Error = ArrowError;

    fn try_from(table: ImmutableTable) -> Result<Self, Self::Error> {
        let ipc_fields = default_ipc_fields(&table.schema.fields);
        let raw_schema = schema_to_bytes(&table.schema(), &ipc_fields);
        let raw_batches = to_raw_batches(table.batches)?;

        Ok(RawTable {
            name: table.name,
            schema: raw_schema,
            batches: raw_batches,
        })
    }
}

#[allow(unused)]
pub fn write_arrow_file(path: impl AsRef<Path>, table: ImmutableTable) -> Result<(), ArrowError> {
    let file = File::create(path)?;

    let options = ipc::write::WriteOptions { compression: None };
    let mut writer = ipc::write::FileWriter::try_new(file, &table.schema, None, options)?;

    for batch in table.batches {
        writer.write(&batch.chunk, None)?;
    }
    writer.finish()?;
    Ok(())
}

#[allow(unused)]
pub fn arrow_file_reader(
    path: impl AsRef<Path>,
) -> Result<ipc::read::FileReader<File>, ArrowError> {
    let mut file = File::open(path)?;
    let metadata = read_file_metadata(&mut file)?;

    let schema = metadata.schema.clone();

    Ok(ipc::read::FileReader::new(file, metadata, None))
}

#[allow(unused)]
pub fn write_parquet_file(
    path: impl AsRef<Path>,
    table: ImmutableTable,
    compression: bool,
) -> Result<(), parquet::read::ParquetError> {
    let mut writer = File::create(path)?;
    let options = if compression {
        parquet::write::WriteOptions {
            write_statistics: true,
            compression: parquet::write::Compression::Zstd,
            version: parquet::write::Version::V2,
        }
    } else {
        parquet::write::WriteOptions {
            write_statistics: true,
            compression: parquet::write::Compression::Uncompressed,
            version: parquet::write::Version::V2,
        }
    };

    let schema = table.schema;
    let chunks = table.batches.into_iter().map(|a| Ok(a.chunk));
    let encodings = vec![];
    let row_groups =
        parquet::write::RowGroupIterator::try_new(chunks, &schema, options, encodings)?;
    let parquet_schema = row_groups.parquet_schema().to_owned();

    parquet::write::write_file(
        &mut writer,
        row_groups,
        &schema,
        parquet_schema,
        options,
        None,
    );
    Ok(())
}

#[allow(unused)]
pub fn parquet_arrow_reader(
    path: impl AsRef<Path>,
) -> Result<parquet::read::RecordReader<File>, parquet::read::ParquetError> {
    let reader = File::open(path)?;
    let rr = RecordReader::try_new(reader, None, None, None, None)?;
    Ok(rr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ToArrow;
    use tempfile::tempdir;

    #[derive(Arrow, Clone)]
    pub struct Event {
        pub id: u64,
        pub data: f32,
    }

    fn test_table() -> MutableTable {
        let mut table = Event::table();
        let events = 1548;
        for i in 0..events {
            let event = Event {
                id: i as u64,
                data: 1.0,
            };
            table.append(event, None).unwrap();
        }
        table
    }

    #[test]
    fn arrow_file_test() {
        let table = test_table();
        let immutable = table.immutable().unwrap();
        let total_rows = immutable.total_rows();
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("arrow_write");
        let reader_path = file_path.clone();
        // verify write
        assert!(write_arrow_file(file_path, immutable).is_ok());

        // verify rows
        let reader = arrow_file_reader(reader_path).unwrap();
        let rows: usize = reader.map(|r| r.unwrap().len()).sum();
        assert_eq!(rows, total_rows);
    }
    #[test]
    fn parquet_file_test() {
        let table = test_table();
        let immutable = table.immutable().unwrap();
        let schema = immutable.schema();
        let total_rows = immutable.total_rows();
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("parquet_write");
        let reader_path = file_path.clone();

        // verify write
        assert!(write_parquet_file(file_path, immutable, true).is_ok());

        // verify schema
        let mut reader = parquet_arrow_reader(reader_path).unwrap();
        let reader_schema = reader.schema().to_owned();
        assert_eq!(schema, reader_schema);

        // verify rows
        let batch = reader.next().unwrap().unwrap();
        assert_eq!(batch.len(), total_rows);
    }

    #[test]
    fn table_serde_test() {
        let table = test_table();
        let immutable: ImmutableTable = table.immutable().unwrap();
        let total_rows = immutable.total_rows();
        let raw_table: RawTable = RawTable::try_from(immutable).unwrap();
        let back_to_immutable: ImmutableTable = ImmutableTable::try_from(raw_table).unwrap();
        assert_eq!(back_to_immutable.total_rows(), total_rows);
    }
}
