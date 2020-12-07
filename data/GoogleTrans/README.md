## Aggregation of Parallel Sentences using Google Translate

We perform English to Arabic and Chinese translations using Google Cloud API (v2). During translation, we use special symbol to identify subject and object in the sentence.

**Examples**

```
English sentence: her <i> stockbroker </i> was also <b> charged </b> .
Chinese translation: 她的<i>股票经纪人</i>也<b>被起诉</b> 。
Arabic translation: كما <b>اتهم</b> <i>سمسار الأوراق المالية</i> لها.
```

We use `<b> </b>` and `<i> </i>` to mark subject and object, respectively. Note, the definition of subject and object differs in event and relation extraction. The above example is chosen from event extraction task. So, the subject and object refers to the event trigger and an event argument, respectively.


### Requirements

Make sure the Google Translate API is installed. It can be installed via pip as follows.

```
pip install --upgrade google-cloud-translate
```

### Basic Setup

You need to setup a project and get the credentials in a json file which looks like follows.

```
{
  "type": "service_account",
  "project_id": "ardent-time-272211",
  "private_key_id": "",
  "private_key": "",
  "client_email": "neural-machine-translation@ardent-time-272211.iam.gserviceaccount.com",
  "client_id": "",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/neural-machine-translation%40ardent-time-272211.iam.gserviceaccount.com"
}
```

I removed the `private_key_id`, `private_key`, and `client_id`. For a quickstart, follow [this](https://cloud.google.com/translate/docs/basic/setup-basic). Upon getting the credentials file, put it in the current directory and rename it as `google_cloud_credentials.json`. Otherwise adjust the code [here](https://github.com/wasiahmad/GATE/blob/main/data/GoogleTrans/translate.py#L12).

Once the project setup is done, the full processing can be exeucted by running the [bash script](https://github.com/wasiahmad/GATE/blob/main//data/GoogleTrans/setup.sh) as follows.

```
$ bash setup.sh
```


### Notes/Tips/Tricks

- We used one-time signup credit (of $300) to use Google Translate services.
- We performed ~4,200 translations and it costs us approx. $20.
- Translation can be performed in batches, however, we did each translation individually.
  - Check the example `translate_text` function [here](https://github.com/googleapis/python-translate/blob/master/samples/snippets/snippets.py#L104).
- We may encounter issue because of quota or limit. So, during translation, we ask the program to sleep for 60 seconds after performing a certain number of translation tasks. Check the code [here](https://github.com/wasiahmad/GATE/blob/main/data/GoogleTrans/translate.py#L46).


### Useful resources

- Quickstart [[url]](https://cloud.google.com/translate/docs/basic/setup-basic)
- Cloud API documentation [[url]](https://googleapis.dev/python/translation/latest/index.html)
- Coding example [[url]](https://cloud.google.com/translate/docs/basic/translating-text#translating_text)
- Quotas and limits [[url]](https://cloud.google.com/translate/quotas)
- Check cost and payment history [[url]](https://cloud.google.com/billing/docs/how-to/view-history)


