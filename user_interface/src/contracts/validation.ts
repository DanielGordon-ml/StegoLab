import { Validator, type Schema } from '@cfworker/json-schema';
import configuration_profile from '../../../contracts/entities/ConfigurationProfile.json';
import configuration_update from '../../../contracts/entities/ConfigurationUpdate.json';
import configuration_reset from '../../../contracts/entities/ConfigurationReset.json';
import capabilities from '../../../contracts/entities/Capabilities.json';
import health_status from '../../../contracts/entities/HealthStatus.json';
import error_envelope from '../../../contracts/entities/ErrorEnvelope.json';
import uploaded_image from '../../../contracts/entities/UploadedImage.json';
import capacity_result from '../../../contracts/entities/CapacityResult.json';
import encoding_job_request from '../../../contracts/entities/EncodingJobRequest.json';
import decoding_job_request from '../../../contracts/entities/DecodingJobRequest.json';
import encoding_job_result from '../../../contracts/entities/EncodingJobResult.json';
import decoding_job_result from '../../../contracts/entities/DecodingJobResult.json';
import decoded_text from '../../../contracts/entities/DecodedText.json';
import model_list from '../../../contracts/entities/ModelList.json';
import installed_model from '../../../contracts/entities/InstalledModel.json';
import type {
  ConfigurationProfile,
  ConfigurationReset,
  ConfigurationUpdate,
} from './configuration';
import type { Capabilities, HealthStatus } from './capabilities';
import type { ErrorEnvelope } from './errors';
import type { InstalledModel, ModelList } from './models';
import type {
  CapacityResult,
  DecodedText,
  DecodingJobRequest,
  DecodingJobResult,
  EncodingJobRequest,
  EncodingJobResult,
  UploadedImage,
} from './inference';

export type ValidateFunction<Result> = (value: unknown) => value is Result;

/** Interpret backend schemas without runtime code generation or unsafe-eval. */
export function create_validator<Result>(
  schema: object,
): ValidateFunction<Result> {
  const validator = new Validator(schema as Schema, '2020-12');
  return (value: unknown): value is Result => {
    try {
      return validator.validate(value).valid;
    } catch {
      return false;
    }
  };
}

export const validate_configuration = create_validator<ConfigurationProfile>(
  configuration_profile,
);
export const validate_update =
  create_validator<ConfigurationUpdate>(configuration_update);
export const validate_reset =
  create_validator<ConfigurationReset>(configuration_reset);
export const validate_capabilities =
  create_validator<Capabilities>(capabilities);
export const validate_health = create_validator<HealthStatus>(health_status);
export const validate_error = create_validator<ErrorEnvelope>(error_envelope);
export const validate_uploaded_image =
  create_validator<UploadedImage>(uploaded_image);
export const validate_capacity_result =
  create_validator<CapacityResult>(capacity_result);
export const validate_encoding_request =
  create_validator<EncodingJobRequest>(encoding_job_request);
export const validate_decoding_request =
  create_validator<DecodingJobRequest>(decoding_job_request);
export const validate_encoding_result =
  create_validator<EncodingJobResult>(encoding_job_result);
export const validate_decoding_result =
  create_validator<DecodingJobResult>(decoding_job_result);
export const validate_decoded_text =
  create_validator<DecodedText>(decoded_text);
export const validate_model_list = create_validator<ModelList>(model_list);
export const validate_installed_model =
  create_validator<InstalledModel>(installed_model);
