import sys
import boto3
import base64
import json

def readFileAsBase64(file_path):
    try:
        with open(file_path, "rb") as image_file:
            input_image = base64.b64encode(image_file.read()).decode("utf8")
        return input_image
    except:
        print("bad file name")
        sys.exit(0)


def construct_bedrock_body(base64_string):
    return json.dumps(
        {
            "inputImage": base64_string,
            "embeddingConfig": {"outputEmbeddingLength": 1024},
        }
    )


def get_embedding_from_titan_multimodal(type, input_body):
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime", region_name="us-west-2"
    )
    output_embedding_length = 1024
    if type == "image":
        body = json.dumps({
            "inputImage": input_body,
            "embeddingConfig": {
                "outputEmbeddingLength": output_embedding_length
            }
        })
    elif type == "text":
        body = json.dumps({
            "inputText": input_body,
            "embeddingConfig": {
                "outputEmbeddingLength": output_embedding_length
            }
        })
    else:
        print("Invalid input type.")

    response = bedrock_runtime.invoke_model(
        body=body,
        modelId="amazon.titan-embed-image-v1",
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response.get("body").read())
    # print(response_body)
    return response_body["embedding"]