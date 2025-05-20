from google.cloud import aiplatform


def deploy_model():
    aiplatform.init(project="YOUR_PROJECT", location="us-central1")
    model = aiplatform.Model.upload(
        display_name="taxi-classifier",
        artifact_uri="gs://YOUR_BUCKET/models/taxi/",
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.0-24:latest",
    )
    endpoint = model.deploy(machine_type="n1-standard-2")
    print("Model deployed to:", endpoint.resource_name)
