var __index = {"config":{"lang":["en"],"separator":"[\\s\\-]+","pipeline":["stopWordFilter"]},"docs":[{"location":"index.html","title":"Welcome to MkDocs","text":"<p>For full documentation visit mkdocs.org.</p>"},{"location":"index.html#commands","title":"Commands","text":"<ul> <li><code>mkdocs new [dir-name]</code> - Create a new project.</li> <li><code>mkdocs serve</code> - Start the live-reloading docs server.</li> <li><code>mkdocs build</code> - Build the documentation site.</li> <li><code>mkdocs -h</code> - Print help message and exit.</li> </ul>"},{"location":"index.html#project-layout","title":"Project layout","text":"<pre><code>mkdocs.yml    # The configuration file.\ndocs/\n    index.md  # The documentation homepage.\n    ...       # Other markdown pages, images and other files.\n</code></pre>"},{"location":"index.html#katex","title":"katex","text":"<p>$\\frac{1}{2}$</p>"},{"location":"about.html","title":"about","text":""},{"location":"gcp/pubsub.html","title":"pubsub","text":""},{"location":"gcp/pubsub.html#_1","title":"\u7528\u8a9e\u3092\u307e\u3068\u3081\u308b","text":"<p>\u767b\u5834\u4eba\u7269\u306fPublisher, Topic, Subscription, Subscriber</p> <p>\ud83d\udcd5\u00a0\u30c8\u30d4\u30c3\u30af</p> <p>Publisher(\u30e1\u30c3\u30bb\u30fc\u30b8\u306e\u9001\u4fe1\u5143)\u30681:1\u3067\u5bfe\u5fdc\u3057\u3066\u3044\u308b\u3082\u306e</p> <p>\ud83d\udcd5\u00a0\u30b5\u30d6\u30b9\u30af\u30ea\u30d7\u30b7\u30e7\u30f3</p> <p>\u5c4a\u3044\u305f\u30e1\u30c3\u30bb\u30fc\u30b8\u306e\u7ba1\u7406\u3001\u30b5\u30d6\u30b9\u30af\u30e9\u30a4\u30d0\u30fc(\u53d7\u4fe1\u8005)\u306b\u30e1\u30c3\u30bb\u30fc\u30b8\u3092\u9001\u308b</p>"},{"location":"gcp/pubsub.html#_2","title":"\u30ed\u30fc\u30ab\u30eb\u3067\u306e\u52d5\u4f5c\u78ba\u8a8d\u65b9\u6cd5\u7b49","text":"<p>pull\u578b\u3068push\u578b\u304c\u5b58\u5728\u3057\u3066\u304a\u308a\u3001pull\u578b\u306f\u53d7\u4fe1\u8005\u304c\u30c7\u30fc\u30bf\u3092\u53d6\u308a\u306b\u3044\u304f\u3001push\u578b\u306f\u30b5\u30d6\u30b9\u30af\u30ea\u30d7\u30b7\u30e7\u30f3\u304c\u30c7\u30fc\u30bf\u3092\u5c4a\u3051\u308b\u30a4\u30e1\u30fc\u30b8</p> <p>\u30ed\u30fc\u30ab\u30eb\u3067\u78ba\u304b\u3081\u308b</p> <pre><code>gcloud pubsub topics create tosa-sample\ngcloud pubsub subscriptions create tosa-sample-sub --topic=tosa-sample\ngcloud pubsub topics publish tosa-sample --message=\"hello\"\ngcloud pubsub subscriptions pull tosa-sample-sub --auto-ack\n</code></pre> <p>\u4e0a\u8a18\u306e\u30b3\u30de\u30f3\u30c9\u306f\u3001Google Cloud Platform\uff08GCP\uff09\u306ePub/Sub\uff08\u30d1\u30d6\u30ea\u30c3\u30b7\u30e5/\u30b5\u30d6\u30b9\u30af\u30e9\u30a4\u30d6\uff09\u30b5\u30fc\u30d3\u30b9\u3092\u4f7f\u7528\u3057\u3066\u3001\u30e1\u30c3\u30bb\u30fc\u30b8\u30f3\u30b0\u30b7\u30b9\u30c6\u30e0\u3092\u30bb\u30c3\u30c8\u30a2\u30c3\u30d7\u3057\u3001\u30e1\u30c3\u30bb\u30fc\u30b8\u3092\u9001\u53d7\u4fe1\u3059\u308b\u305f\u3081\u306e\u3082\u306e\u3067\u3059\u3002\u4ee5\u4e0b\u3001\u305d\u308c\u305e\u308c\u306e\u30b3\u30de\u30f3\u30c9\u3092\u89e3\u8aac\u3057\u307e\u3059\u3002</p> <ol> <li><code>gcloud pubsub topics create tosa-sample</code></li> </ol> <p>\u3053\u306e\u30b3\u30de\u30f3\u30c9\u306f\u3001Pub/Sub\u30c8\u30d4\u30c3\u30af\u3092\u4f5c\u6210\u3057\u307e\u3059\u3002\u30c8\u30d4\u30c3\u30af\u306f\u30e1\u30c3\u30bb\u30fc\u30b8\u306e\u9001\u4fe1\u5143\u3067\u3042\u308a\u3001\u30e1\u30c3\u30bb\u30fc\u30b8\u306e\u9001\u4fe1\u5148\u3092\u8b58\u5225\u3057\u307e\u3059\u3002\u3053\u306e\u30b3\u30de\u30f3\u30c9\u3067\u306f\u3001\"tosa-sample\"\u3068\u3044\u3046\u540d\u524d\u306e\u30c8\u30d4\u30c3\u30af\u3092\u4f5c\u6210\u3057\u3066\u3044\u307e\u3059\u3002</p> <ol> <li><code>gcloud pubsub subscriptions create tosa-sample-sub --topic=tosa-sample</code></li> </ol> <p>\u3053\u306e\u30b3\u30de\u30f3\u30c9\u306f\u3001\u30c8\u30d4\u30c3\u30af\u306b\u5bfe\u3059\u308b\u30b5\u30d6\u30b9\u30af\u30ea\u30d7\u30b7\u30e7\u30f3\uff08\u8cfc\u8aad\uff09\u3092\u4f5c\u6210\u3057\u307e\u3059\u3002\u30b5\u30d6\u30b9\u30af\u30ea\u30d7\u30b7\u30e7\u30f3\u306f\u3001\u30c8\u30d4\u30c3\u30af\u306b\u5bfe\u3057\u3066\u30e1\u30c3\u30bb\u30fc\u30b8\u3092\u53d7\u4fe1\u3059\u308b\u5bfe\u8c61\u3092\u6307\u5b9a\u3057\u307e\u3059\u3002\u3053\u3053\u3067\u306f\u3001\"tosa-sample-sub\"\u3068\u3044\u3046\u540d\u524d\u306e\u30b5\u30d6\u30b9\u30af\u30ea\u30d7\u30b7\u30e7\u30f3\u3092\u3001\u5148\u307b\u3069\u4f5c\u6210\u3057\u305f\"tosa-sample\"\u30c8\u30d4\u30c3\u30af\u306b\u5bfe\u3057\u3066\u4f5c\u6210\u3057\u3066\u3044\u307e\u3059\u3002</p> <ol> <li><code>gcloud pubsub topics publish tosa-sample --message=\"hello\"</code></li> </ol> <p>\u3053\u306e\u30b3\u30de\u30f3\u30c9\u306f\u3001\u6307\u5b9a\u3057\u305f\u30c8\u30d4\u30c3\u30af\uff08\"tosa-sample\"\uff09\u306b\u30e1\u30c3\u30bb\u30fc\u30b8\u3092\u516c\u958b\uff08\u9001\u4fe1\uff09\u3057\u307e\u3059\u3002\u30e1\u30c3\u30bb\u30fc\u30b8\u306e\u5185\u5bb9\u306f\u3001\"--message\"\u30d5\u30e9\u30b0\u3067\u6307\u5b9a\u3055\u308c\u3066\u304a\u308a\u3001\u3053\u3053\u3067\u306f\"hello\"\u3068\u3044\u3046\u6587\u5b57\u5217\u3092\u542b\u3093\u3067\u3044\u307e\u3059\u3002</p> <ol> <li><code>gcloud pubsub subscriptions pull tosa-sample-sub --auto-ack</code></li> </ol> <p>\u3053\u306e\u30b3\u30de\u30f3\u30c9\u306f\u3001\u6307\u5b9a\u3057\u305f\u30b5\u30d6\u30b9\u30af\u30ea\u30d7\u30b7\u30e7\u30f3\uff08\"tosa-sample-sub\"\uff09\u304b\u3089\u30e1\u30c3\u30bb\u30fc\u30b8\u3092\u53d6\u5f97\uff08\u30d7\u30eb\uff09\u3057\u307e\u3059\u3002\"--auto-ack\"\u30d5\u30e9\u30b0\u304c\u4f7f\u7528\u3055\u308c\u3066\u3044\u308b\u305f\u3081\u3001\u30e1\u30c3\u30bb\u30fc\u30b8\u3092\u53d7\u4fe1\u3057\u305f\u5f8c\u3001\u660e\u793a\u7684\u306a\u78ba\u8a8d\u3092\u884c\u308f\u305a\u306b\u81ea\u52d5\u7684\u306b\u53d7\u4fe1\u304c\u78ba\u8a8d\u3055\u308c\u307e\u3059\uff08ACK\u304c\u9001\u4fe1\u3055\u308c\u307e\u3059\uff09\u3002\u3053\u308c\u306b\u3088\u308a\u3001\u30e1\u30c3\u30bb\u30fc\u30b8\u304c\u4e00\u5ea6\u3057\u304b\u51e6\u7406\u3055\u308c\u306a\u3044\u3088\u3046\u306b\u306a\u308a\u307e\u3059\u3002</p> <p>\u3053\u308c\u3089\u306e\u30b3\u30de\u30f3\u30c9\u3092\u9806\u306b\u5b9f\u884c\u3059\u308b\u3068\u3001\"tosa-sample\"\u30c8\u30d4\u30c3\u30af\u306b\"hello\"\u3068\u3044\u3046\u30e1\u30c3\u30bb\u30fc\u30b8\u304c\u9001\u4fe1\u3055\u308c\u3001\"tosa-sample-sub\"\u30b5\u30d6\u30b9\u30af\u30ea\u30d7\u30b7\u30e7\u30f3\u304c\u305d\u306e\u30e1\u30c3\u30bb\u30fc\u30b8\u3092\u53d7\u4fe1\u3057\u3066\u51e6\u7406\u3059\u308b\u3053\u3068\u306b\u306a\u308a\u307e\u3059\u3002\u305f\u3060\u3057\u3001\u5b9f\u969b\u306b\u306f\u3001\u30e1\u30c3\u30bb\u30fc\u30b8\u3092\u53d7\u4fe1\u3057\u305f\u5f8c\u306e\u51e6\u7406\u3084\u3001\u30b5\u30d6\u30b9\u30af\u30ea\u30d7\u30b7\u30e7\u30f3\u306e\u8a2d\u5b9a\u306a\u3069\u3001\u3055\u307e\u3056\u307e\u306a\u5074\u9762\u304c\u5b58\u5728\u3057\u307e\u3059\u3002</p>"}]}