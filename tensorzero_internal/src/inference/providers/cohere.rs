//! Cohere API provider implementation.

use super::openai::{prepare_openai_messages, OpenAIRequestMessage, OpenAITool, OpenAIToolChoice};
use crate::endpoints::inference::InferenceCredentials;
use crate::error::{Error, ErrorDetails};
use crate::inference::providers::provider_trait::InferenceProvider;
use crate::inference::types::{ModelInferenceRequest, ProviderInferenceResponse};
use crate::model::Credential;
use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use tokio::time::Instant;

#[derive(Debug)]
pub struct CohereProvider {
    model_name: String,
    credentials: CohereCredentials,
}
#[derive(Debug, Serialize, Deserialize)]
struct CohereResponse {
    id: String,
    generations: Vec<CohereGeneration>,
    message: String,
    metadata: CohereMeta,
}

#[derive(Debug, Serialize)]
struct CohereGeneration {
    id: String,
    text: String,
    finish_reason: String,
    tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
struct CohereMeta {
    api_version: String,
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Clone)]
pub enum CohereCredentials {
    Static(SecretString),
    Dynamic(String),
    None,
}
pub enum CohereSafetyMode {
    CONTEXTUAL,
    OFF,
    STRICT,
}
pub enum CohereCitationOptions {
    FAST,
    ACCURATE,
    OFF,
}
#[derive(Debug, Serialize)]
struct CohereRequest<'a> {
    messages: Vec<OpenAIRequestMessage<'a>>,
    model: &'a str,
    stream: Option<bool>,

    //#[serde(skip_serializing_if = "Option::is_none")]
    //respone_format: OptionFuture<CohereResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequence: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    k: Option<f32>, //TODO: [Alberto] max == 500 keep in mind
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprop: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAIToolChoice<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict_tools: Option<bool>, // TODO: [Alberto] This is still on beta
}

impl<'a> CohereRequest<'a> {
    pub fn new(model: &'a str, request: &'a ModelInferenceRequest<'_>) -> Result<Self, Error> {
        let ModelInferenceRequest {
            temperature,
            max_tokens,
            seed,
            top_p,
            frequency_penalty,
            stream,
            ..
        } = *request;
        let messages = prepare_openai_messages(request);
        Ok(CohereRequest {
            messages,
            model,
            stream: Some(stream),
            temperature,
            max_tokens,
            seed,
            top_p,
            frequency_penalty,
            logprop: None,
            tools: None,
            tool_choice: None,
            strict_tools: None,
            k: None,
            presence_penalty: None,
            stop_sequence: None,
        })
    }
}

impl TryFrom<Credential> for CohereCredentials {
    type Error = Error;

    fn try_from(credentials: Credential) -> Result<Self, Error> {
        match credentials {
            Credential::Static(key) => Ok(CohereCredentials::Static(key)),
            Credential::Dynamic(key_name) => Ok(CohereCredentials::Dynamic(key_name)),
            Credential::None => Ok(CohereCredentials::None),
            #[cfg(any(test, feature = "e2e_tests"))]
            Credential::Missing => Ok(CohereCredentials::None),
            _ => Err(Error::new(ErrorDetails::Config {
                message: "Invalid api_key_location for Cohere provider".to_string(),
            })),
        }
    }
}

impl CohereCredentials {
    pub fn get_api_key<'a>(
        &'a self,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<&'a SecretString, Error> {
        match self {
            CohereCredentials::Static(api_key) => Ok(api_key),
            CohereCredentials::Dynamic(key_name) => {
                dynamic_api_keys.get(key_name).ok_or_else(|| {
                    ErrorDetails::ApiKeyMissing {
                        provider_name: "Cohere".to_string(),
                    }
                    .into()
                })
            }
            CohereCredentials::None => Err(ErrorDetails::ApiKeyMissing {
                provider_name: "Cohere".to_string(),
            }
            .into()),
        }
    }
}

impl InferenceProvider for CohereProvider {
    async fn infer<'a>(
        &'a self,
        request: &'a ModelInferenceRequest<'_>,
        http_client: &'a reqwest::Client,
        dynamic_api_keys: &'a InferenceCredentials,
    ) -> Result<u32, Error> {
        // TODO: return response
        let request_body = CohereRequest::new(&self.model_name, request)?;
        let api_key = self.credentials.get_api_key(dynamic_api_keys)?;
        let start_time = Instant::now();
        // Create the request
        let request_builder = http_client
            .post("<cohere-api....>") //TODO: fix this
            .bearer_auth(api_key.expose_secret());
        // Send the request
        let res = request_builder
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                //if the request fails, return an error
                Error::new(ErrorDetails::InferenceClient {
                    message: format!("Error sending request to Cohere: {e}"),
                    status_code: e.status(),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: "cohere".to_string(),
                })
            })?;
        // Check if the request was successful
        if res.status().is_success() {
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing text response: {e}"),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: "cohere".to_string(),
                })
            })?;
            // Assert the response is a valid json.
            let response = serde_json::from_str::<CohereResponse>(&response).map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsing Json response: {e}: {response}"),
                    raw_request: Some(serde_json::to_string(&request_body).unwrap_or_default()),
                    raw_response: None,
                    provider_type: "cohere".to_string(),
                })
            })?;

            //Ok() TODO: return response
            Ok(0)
        } else {
            let status_code = res.status();
            let response = res.text().await.map_err(|e| {
                Error::new(ErrorDetails::InferenceServer {
                    message: format!("Error parsisng error response: {e}"),
                    raw_request: Some(res.text().await.unwrap_or_default()),
                    raw_response: None,
                    provider_type: "cohere".to_string(),
                })
            })?;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::inference::types::{ModelInferenceRequest, RequestMessage};

    use super::*;

    use uuid::Uuid;

    #[test]
    fn text_cohere_equest_new() {
        let request = ModelInferenceRequest {
            inference_id: Uuid::new_v7(),
            messages: vec![RequestMessage {
                role: Role::User,
                content: vec!["What's the weather like?".to_string().into()],
            }],
            system: None,
            temperature: Some(0.5),
            top_p: None,
            frequency_penalty: None,
            max_tokens,
        };
        assert!(true)
    }
}
