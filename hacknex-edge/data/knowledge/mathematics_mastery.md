# Mathematics Mastery: Comprehensive Formulas, Theories, and Step-by-Step Problem Solving Guide
HackNEX 2026 — EDRA Edge AI Brain Ground-Truth Knowledge Base

---

## 1. Arithmetic & Number Theory

- **Basic Operations & Order of Operations**: PEMDAS / BODMAS (Parentheses/Brackets, Exponents/Orders, Multiplication & Division from left to right, Addition & Subtraction from left to right).
- **Divisibility Rules**:
  - Divisible by 2: Last digit is even (0, 2, 4, 6, 8).
  - Divisible by 3: Sum of all digits is divisible by 3.
  - Divisible by 4: Last two digits form a number divisible by 4.
  - Divisible by 5: Last digit is 0 or 5.
  - Divisible by 6: Divisible by both 2 and 3.
  - Divisible by 8: Last three digits form a number divisible by 8.
  - Divisible by 9: Sum of all digits is divisible by 9.
  - Divisible by 11: Alternating sum of digits is a multiple of 11 (or 0).
- **Greatest Common Divisor (GCD) & Least Common Multiple (LCM)**:
  - Euclidean Algorithm: $\text{gcd}(a, b) = \text{gcd}(b, a \pmod b)$ until remainder is 0.
  - Relationship: $a \times b = \text{gcd}(a, b) \times \text{lcm}(a, b)$.
- **Sequences and Progressions**:
  - **Arithmetic Progression (AP)**:
    - $n$-th term: $a_n = a + (n-1)d$
    - Sum of first $n$ terms: $S_n = \frac{n}{2}[2a + (n-1)d] = \frac{n}{2}(a + l)$ where $l$ is the last term.
  - **Geometric Progression (GP)**:
    - $n$-th term: $a_n = a \cdot r^{n-1}$
    - Sum of first $n$ terms: $S_n = a \frac{r^n - 1}{r - 1}$ for $r \neq 1$.
    - Infinite GP sum (when $|r| < 1$): $S_\infty = \frac{a}{1 - r}$.

---

## 2. Algebra, Polynomials & Equations

- **Quadratic Equations**:
  - Standard form: $ax^2 + bx + c = 0$ ($a \neq 0$).
  - Discriminant: $\Delta = b^2 - 4ac$.
    - If $\Delta > 0$: Two distinct real roots.
    - If $\Delta = 0$: Exactly one real root (repeated root).
    - If $\Delta < 0$: Two complex conjugate roots.
  - Quadratic Formula: $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$.
  - Vieta's Formulas: Sum of roots $\alpha + \beta = -\frac{b}{a}$, Product of roots $\alpha \beta = \frac{c}{a}$.
- **Cubic and Higher Order Polynomials**:
  - Cubic form: $ax^3 + bx^2 + cx + d = 0$.
  - Vieta's formulas for cubic: $\sum \alpha = -b/a$, $\sum \alpha\beta = c/a$, $\alpha\beta\gamma = -d/a$.
- **Exponents and Logarithms**:
  - Laws of exponents: $a^m \cdot a^n = a^{m+n}$, $\frac{a^m}{a^n} = a^{m-n}$, $(a^m)^n = a^{mn}$, $a^{-n} = \frac{1}{a^n}$, $a^0 = 1$.
  - Laws of logarithms:
    - $\log_b(xy) = \log_b(x) + \log_b(y)$
    - $\log_b(x/y) = \log_b(x) - \log_b(y)$
    - $\log_b(x^k) = k \log_b(x)$
    - Change of base formula: $\log_b(x) = \frac{\ln x}{\ln b} = \frac{\log_{10} x}{\log_{10} b}$
    - $\ln(e) = 1$, $\ln(1) = 0$.
- **Binomial Theorem**:
  - $(a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k$, where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$.

---

## 3. Trigonometry

- **Definitions in a Right Triangle**:
  - $\sin \theta = \frac{\text{Opposite}}{\text{Hypotenuse}}$, $\cos \theta = \frac{\text{Adjacent}}{\text{Hypotenuse}}$, $\tan \theta = \frac{\text{Opposite}}{\text{Adjacent}} = \frac{\sin \theta}{\cos \theta}$.
  - Reciprocals: $\csc \theta = \frac{1}{\sin \theta}$, $\sec \theta = \frac{1}{\cos \theta}$, $\cot \theta = \frac{1}{\tan \theta}$.
- **Fundamental Pythagorean Identities**:
  - $\sin^2 \theta + \cos^2 \theta = 1$
  - $1 + \tan^2 \theta = \sec^2 \theta$
  - $1 + \cot^2 \theta = \csc^2 \theta$
- **Angle Addition and Subtraction**:
  - $\sin(A \pm B) = \sin A \cos B \pm \cos A \sin B$
  - $\cos(A \pm B) = \cos A \cos B \mp \sin A \sin B$
  - $\tan(A \pm B) = \frac{\tan A \pm \tan B}{1 \mp \tan A \tan B}$
- **Double Angle and Half Angle Formulas**:
  - $\sin(2\theta) = 2 \sin \theta \cos \theta$
  - $\cos(2\theta) = \cos^2 \theta - \sin^2 \theta = 2\cos^2 \theta - 1 = 1 - 2\sin^2 \theta$
  - $\tan(2\theta) = \frac{2\tan \theta}{1 - \tan^2 \theta}$
- **Laws of Sines and Cosines**:
  - Law of Sines: $\frac{a}{\sin A} = \frac{b}{\sin B} = \frac{c}{\sin C} = 2R$.
  - Law of Cosines: $c^2 = a^2 + b^2 - 2ab \cos C$.

---

## 4. Differential and Integral Calculus

- **Limits**:
  - L'Hôpital's Rule: If $\lim_{x \to c} \frac{f(x)}{g(x)}$ gives indeterminate form $\frac{0}{0}$ or $\frac{\pm \infty}{\pm \infty}$, then $\lim_{x \to c} \frac{f(x)}{g(x)} = \lim_{x \to c} \frac{f'(x)}{g'(x)}$.
  - Standard limits: $\lim_{x \to 0} \frac{\sin x}{x} = 1$, $\lim_{x \to 0} \frac{e^x - 1}{x} = 1$, $\lim_{x \to \infty} (1 + \frac{1}{x})^x = e$.
- **Derivatives**:
  - Power rule: $\frac{d}{dx}[x^n] = n x^{n-1}$.
  - Product rule: $\frac{d}{dx}[uv] = u'v + uv'$.
  - Quotient rule: $\frac{d}{dx}[\frac{u}{v}] = \frac{u'v - uv'}{v^2}$.
  - Chain rule: $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$.
  - Key derivatives: $\frac{d}{dx}[e^x] = e^x$, $\frac{d}{dx}[\ln x] = \frac{1}{x}$, $\frac{d}{dx}[\sin x] = \cos x$, $\frac{d}{dx}[\cos x] = -\sin x$, $\frac{d}{dx}[\tan x] = \sec^2 x$.
- **Integrals**:
  - Power rule: $\int x^n \, dx = \frac{x^{n+1}}{n+1} + C$ ($n \neq -1$).
  - Logarithmic: $\int \frac{1}{x} \, dx = \ln |x| + C$.
  - Exponential: $\int e^{kx} \, dx = \frac{1}{k} e^{kx} + C$.
  - Integration by Parts: $\int u \, dv = uv - \int v \, du$.
  - Substitution method: $\int f(g(x)) g'(x) \, dx = \int f(u) \, du$ where $u = g(x)$.

---

## 5. Linear Algebra & Matrices

- **Matrix Multiplication**: If matrix $A$ has dimension $m \times k$ and matrix $B$ has dimension $k \times n$, then $C = AB$ has dimension $m \times n$, where $C_{ij} = \sum_{r=1}^{k} A_{ir} B_{rj}$.
- **Determinant ($2 \times 2$ and $3 \times 3$)**:
  - For $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$: $\det = ad - bc$.
  - Inverse of $2 \times 2$: $A^{-1} = \frac{1}{ad - bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$.
- **Eigenvalues and Eigenvectors**:
  - For square matrix $A$, vector $v \neq 0$ is an eigenvector with eigenvalue $\lambda$ if $Av = \lambda v$.
  - Characteristic equation: $\det(A - \lambda I) = 0$.

---

## 6. Probability, Combinatorics & Statistics

- **Permutations and Combinations**:
  - Permutations ($n$ objects taken $r$ at a time, order matters): $P(n, r) = \frac{n!}{(n-r)!}$.
  - Combinations ($n$ objects taken $r$ at a time, order does not matter): $\binom{n}{r} = \frac{n!}{r!(n-r)!}$.
- **Probability Rules**:
  - Addition rule: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$.
  - Conditional probability: $P(A | B) = \frac{P(A \cap B)}{P(B)}$.
  - Bayes' Theorem: $P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}$.
- **Descriptive Statistics**:
  - Mean: $\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$.
  - Variance: $\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$.
  - Standard Deviation: $\sigma = \sqrt{\text{Variance}}$.
