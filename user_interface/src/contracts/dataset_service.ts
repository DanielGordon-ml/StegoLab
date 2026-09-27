import inspection_request_schema from '../../../contracts/entities/DatasetInspectionRequest.json';
import inspection_schema from '../../../contracts/entities/DatasetInspection.json';
import fetch_request_schema from '../../../contracts/entities/DatasetFetchRequest.json';
import fetch_summary_schema from '../../../contracts/entities/DatasetFetchSummary.json';
import storage_schema from '../../../contracts/entities/DatasetStorageSummary.json';
import { create_validator } from './validation';
import { request_validated } from './transport';
import { validate_job } from './workflow_service';
import { RequestFailure } from './errors';
import type { JobSnapshot } from './workflows';
import type {
  DatasetFetchRequest,
  DatasetFetchSummary,
  DatasetInspection,
  DatasetInspectionRequest,
  DatasetSourceSpec,
  DatasetStorageSummary,
} from './datasets';

/** Inspection may resolve a remote revision, so it waits longer than a plain read. */
export const INSPECTION_TIMEOUT_MILLISECONDS = 30000;

export const validate_inspection_request =
  create_validator<DatasetInspectionRequest>(inspection_request_schema);
export const validate_inspection =
  create_validator<DatasetInspection>(inspection_schema);
export const validate_fetch_request =
  create_validator<DatasetFetchRequest>(fetch_request_schema);
export const validate_fetch_summary =
  create_validator<DatasetFetchSummary>(fetch_summary_schema);
export const validate_storage_summary =
  create_validator<DatasetStorageSummary>(storage_schema);

type SchemaProperties = Record<string, object>;
interface RequestSchema {
  properties: SchemaProperties;
  $defs: Record<string, { properties: SchemaProperties }>;
}
const SOURCE_DEFINITIONS: Record<DatasetSourceSpec['source_kind'], string> = {
  hugging_face: 'HuggingFaceSourceSpec',
  https_archive: 'HttpsArchiveSourceSpec',
  upload: 'UploadSourceSpec',
};

/** Collect the default values a schema declares for its fields. */
function schema_defaults(properties: SchemaProperties) {
  return Object.fromEntries(
    Object.entries(properties)
      .filter(([, property]) => 'default' in property)
      .map(([name, property]) => [
        name,
        'default' in property ? property.default : undefined,
      ]),
  );
}

/** Drop fields a form left undefined so they do not hide contract defaults. */
function defined_fields(value: object) {
  return Object.fromEntries(
    Object.entries(value).filter(([, field]) => field !== undefined),
  );
}

/** Fill the defaults of one source description from the request schema. */
function complete_source(schema: RequestSchema, source: DatasetSourceSpec) {
  const definition = schema.$defs[SOURCE_DEFINITIONS[source.source_kind]];
  return {
    ...schema_defaults(definition ? definition.properties : {}),
    ...defined_fields(source),
  };
}

/** Fill every default the backend declares so partial form values validate. */
function complete_request(
  schema: RequestSchema,
  request: { source: DatasetSourceSpec },
) {
  return {
    ...schema_defaults(schema.properties),
    ...defined_fields(request),
    source: complete_source(schema, request.source),
  };
}

/** Ask the backend what a source contains before anything is downloaded. */
export async function inspect_dataset_source(
  request: DatasetInspectionRequest,
): Promise<DatasetInspection> {
  const normalized = complete_request(inspection_request_schema, request);
  if (!validate_inspection_request(normalized))
    throw new RequestFailure('Check the source details and try again.');
  return request_validated(
    '/datasets/inspections',
    validate_inspection,
    { method: 'POST', body: JSON.stringify(normalized) },
    INSPECTION_TIMEOUT_MILLISECONDS,
  );
}

/** Fill the contract defaults and refuse a request the backend would reject. */
export function prepare_fetch_request(
  request: DatasetFetchRequest,
): DatasetFetchRequest {
  const normalized = complete_request(fetch_request_schema, request);
  if (!validate_fetch_request(normalized))
    throw new RequestFailure(
      'Check the dataset names and source details, then try again.',
    );
  return normalized;
}

/** Start one download-and-prepare job using a stable request identifier. */
export async function start_dataset_fetch(
  request: DatasetFetchRequest,
): Promise<JobSnapshot> {
  const normalized = prepare_fetch_request(request);
  return request_validated('/datasets/fetch_jobs', validate_job, {
    method: 'POST',
    body: JSON.stringify(normalized),
  });
}

/** Read disk use of downloads, raw folders and prepared datasets. */
export const read_dataset_storage = () =>
  request_validated('/datasets/storage', validate_storage_summary);

/** Delete downloaded archives that no raw folder uses any more. */
export const remove_unused_downloads = () =>
  request_validated('/datasets/cache/unused', validate_storage_summary, {
    method: 'DELETE',
  });
