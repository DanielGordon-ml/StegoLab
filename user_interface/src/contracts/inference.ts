/** Safe description of how an uploaded image was checked and prepared. */
export interface ImageSummary {
  source_format: 'JPEG' | 'PNG';
  mode: 'RGB' | 'RGBA';
  pixel_policy: 'prepare_srgb' | 'preserve_stored';
  source_width: number;
  source_height: number;
  prepared_width: number;
  prepared_height: number;
  input_size_bytes: number;
  orientation: number;
  color_policy: 'icc_to_srgb' | 'declared_srgb' | 'assumed_srgb';
}

/** One accepted upload, kept by the backend for at most one day. */
export interface UploadedImage {
  image_reference: string;
  purpose: 'cover' | 'encoded';
  summary: ImageSummary;
  warnings: string[];
  created_at: string;
  expires_at: string;
}

/** Ask how many message bytes one uploaded image carries with one model. */
export interface CapacityRequest {
  image_reference: string;
  model_identifier: string;
}

/** Layout accounting shared with the backend protocol, without model claims. */
export interface PayloadCapacity {
  profile_identifier: 'test_only_v1';
  maximum_message_bytes: number;
  header_bytes: number;
  message_length_bytes: number;
  authentication_bytes: number;
  padding_bytes_at_capacity: number;
  block_count: number;
  frame_bytes: number;
  correction_bytes: number;
  protected_bits: number;
  payload_map_bits: number;
  repeated_bits: number;
  minimum_repetitions: number;
  additional_repetitions: number;
}

/** The message limit for one image and one installed model. */
export interface CapacityResult {
  image_reference: string;
  model_identifier: string;
  profile_identifier: 'test_only_v1';
  width: number;
  height: number;
  maximum_message_bytes: number;
  capacity: PayloadCapacity;
}
