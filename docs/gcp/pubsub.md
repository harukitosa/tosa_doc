## 用語をまとめる

登場人物はPublisher, Topic, Subscription, Subscriber

**📕 トピック**

Publisher(メッセージの送信元)と1:1で対応しているもの

📕 **サブスクリプション**

届いたメッセージの管理、サブスクライバー(受信者)にメッセージを送る

# ローカルでの動作確認方法等

pull型とpush型が存在しており、pull型は受信者がデータを取りにいく、push型はサブスクリプションがデータを届けるイメージ

ローカルで確かめる

```
gcloud pubsub topics create tosa-sample
gcloud pubsub subscriptions create tosa-sample-sub --topic=tosa-sample
gcloud pubsub topics publish tosa-sample --message="hello"
gcloud pubsub subscriptions pull tosa-sample-sub --auto-ack
```

上記のコマンドは、Google Cloud Platform（GCP）のPub/Sub（パブリッシュ/サブスクライブ）サービスを使用して、メッセージングシステムをセットアップし、メッセージを送受信するためのものです。以下、それぞれのコマンドを解説します。

1. **`gcloud pubsub topics create tosa-sample`**

このコマンドは、Pub/Subトピックを作成します。トピックはメッセージの送信元であり、メッセージの送信先を識別します。このコマンドでは、"tosa-sample"という名前のトピックを作成しています。

1. **`gcloud pubsub subscriptions create tosa-sample-sub --topic=tosa-sample`**

このコマンドは、トピックに対するサブスクリプション（購読）を作成します。サブスクリプションは、トピックに対してメッセージを受信する対象を指定します。ここでは、"tosa-sample-sub"という名前のサブスクリプションを、先ほど作成した"tosa-sample"トピックに対して作成しています。

1. **`gcloud pubsub topics publish tosa-sample --message="hello"`**

このコマンドは、指定したトピック（"tosa-sample"）にメッセージを公開（送信）します。メッセージの内容は、"--message"フラグで指定されており、ここでは"hello"という文字列を含んでいます。

1. **`gcloud pubsub subscriptions pull tosa-sample-sub --auto-ack`**

このコマンドは、指定したサブスクリプション（"tosa-sample-sub"）からメッセージを取得（プル）します。"--auto-ack"フラグが使用されているため、メッセージを受信した後、明示的な確認を行わずに自動的に受信が確認されます（ACKが送信されます）。これにより、メッセージが一度しか処理されないようになります。

これらのコマンドを順に実行すると、"tosa-sample"トピックに"hello"というメッセージが送信され、"tosa-sample-sub"サブスクリプションがそのメッセージを受信して処理することになります。ただし、実際には、メッセージを受信した後の処理や、サブスクリプションの設定など、さまざまな側面が存在します。