import apache_beam as beam
from apache_beam import Pipeline
from google.cloud import storage


def tensorize_mri(pipeline: Pipeline, output_file: str):

    # Query bucket in BQ
    # blobs=["projects/pbatra/mri_test/2345370_20211_2_0.zip", "projects/pbatra/mri_test/2345370_20212_2_0.zip"]
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("ml4cvd")
    blobs = bucket.list_blobs(prefix="projects/pbatra/mri_test/")

    # output_file = 'gs://ml4cvd/projects/pbatra/temp/%s.csv' % RUN_NAME
    output_file = '/Users/kyuksel/ml4cvd/tensors/dataflow_tensors/tensors_test_mri'

    all_files = (
                pipeline
                # create a list of files to read
                | 'create_file_path_tuple' >> beam.Create([blob.name for blob in blobs]))

                | 'process_file' >> beam.Map(_write_tensors_from_zipped_dicoms)

                | 'Writing to file %s' % output_file >> beam.io.WriteToText(output_file)
    )

    result = p.run()
    result.wait_until_finish()
