import math, random

def sigmoid(z):
    return 1 / (1 + math.exp(-max(-40, min(40, z))))

def auc(ys, ps):
    pos = [p for y, p in zip(ys, ps) if y == 1]
    neg = [p for y, p in zip(ys, ps) if y == 0]
    return sum(((a > b) + 0.5 * (a == b) for a in pos for b in neg)) / (len(pos) * len(neg))

def run(p):
    rows = p.get('rows')
    seed = p.get('seed', 42)
    if rows is None:
        n = p.get('samples', 400)
        if not isinstance(n, int) or not 80 <= n <= 2000:
            raise ValueError('samples must be 80–2000')
        rng = random.Random(seed)
        rows = []
        for _ in range(n):
            tenure = rng.uniform(1, 60)
            tickets = rng.randrange(0, 9)
            usage = rng.random()
            prob = sigmoid(-0.055 * tenure + 0.45 * tickets - 2 * usage + 0.1)
            rows.append({'tenure': tenure, 'tickets': tickets, 'usage': usage, 'churn': int(rng.random() < prob)})
    features = ['tenure', 'tickets', 'usage']
    if not isinstance(rows, list) or not 40 <= len(rows) <= 2000:
        raise ValueError('rows must contain 40–2000 observations')
    for r in rows:
        if r.get('churn') not in [0, 1] or any((not isinstance(r.get(k), (int, float)) or not math.isfinite(r[k]) for k in features)):
            raise ValueError('Finite numeric features and binary churn required')
    groups = [[r for r in rows if r['churn'] == y] for y in [0, 1]]
    if min(map(len, groups)) < 5:
        raise ValueError('At least 5 observations per class required')
    rng = random.Random(seed)
    train = []
    test = []
    for g in groups:
        rng.shuffle(g)
        cut = max(1, int(len(g) * 0.25))
        test += g[:cut]
        train += g[cut:]
    means = [sum((r[k] for r in train)) / len(train) for k in features]
    std = [max(1e-09, (sum(((r[k] - m) ** 2 for r in train)) / len(train)) ** 0.5) for k, m in zip(features, means)]

    def vec(r):
        return [1] + [(r[k] - m) / s for k, m, s in zip(features, means, std)]
    xs = [vec(r) for r in train]
    w = [0.0] * 4
    for _ in range(600):
        grad = [0.0] * 4
        for x, r in zip(xs, train):
            err = sigmoid(sum((a * b for a, b in zip(w, x)))) - r['churn']
            for j in range(4):
                grad[j] += err * x[j]
        w = [v - 0.1 * (g / len(train) + (0.02 * v if j else 0)) for j, (v, g) in enumerate(zip(w, grad))]
    ps = [sigmoid(sum((a * b for a, b in zip(w, vec(r))))) for r in test]
    ys = [r['churn'] for r in test]
    pred = [int(x >= 0.5) for x in ps]
    tp = sum((y == t == 1 for y, t in zip(ys, pred)))
    tn = sum((y == t == 0 for y, t in zip(ys, pred)))
    fp = sum((y == 0 and t == 1 for y, t in zip(ys, pred)))
    fn = sum((y == 1 and t == 0 for y, t in zip(ys, pred)))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    majority = int(sum((r['churn'] for r in train)) >= len(train) / 2)
    return {'data_source': 'synthetic' if p.get('rows') is None else 'provided', 'train_rows': len(train), 'test_rows': len(test), 'accuracy': (tp + tn) / len(test), 'precision': prec, 'recall': rec, 'f1': 2 * prec * rec / max(prec + rec, 1e-09), 'roc_auc': auc(ys, ps), 'majority_baseline_accuracy': sum((y == majority for y in ys)) / len(ys), 'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}, 'scaler_fit_on': 'training_only', 'coefficients': dict(zip(features, w[1:]))}
