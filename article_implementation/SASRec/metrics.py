def apk(actual, predicted, k = 10):
    predicted_ = predicted
    if len(actual) == 0:
        return 0
    if len(predicted) >= k:
        predicted_ = predicted[:k]

    ans, cnt = 0, 0
    total = min(len(actual), k)
    for i in range(len(predicted_)):
        if predicted_[i] in actual:
            cnt += 1
            ans += cnt / (i + 1)
    return ans / total

def mapk(actual, predicted, k = 10):
    ans = 0
    cnt = 0
    for i in range(min(len(predicted), len(actual))):
        ans += apk(actual[i], predicted[i], k)
        cnt += 1
    return ans / cnt