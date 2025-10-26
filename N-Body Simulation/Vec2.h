#ifndef VEC2_H
#define VEC2_H

#include <cmath>

// Jednostavna 2D vektor klasa za matematičke operacije
class Vec2 {
public:
    double x, y;

    // Konstruktori
    Vec2() : x(0.0), y(0.0) {}
    Vec2(double x, double y) : x(x), y(y) {}

    // Statička metoda za nula vektor
    static Vec2 zero() { return Vec2(0.0, 0.0); }

    // Sabiranje vektora
    Vec2 operator+(const Vec2& other) const {
        return Vec2(x + other.x, y + other.y);
    }

    // Oduzimanje vektora
    Vec2 operator-(const Vec2& other) const {
        return Vec2(x - other.x, y - other.y);
    }

    // Množenje sa skalarom
    Vec2 operator*(double scalar) const {
        return Vec2(x * scalar, y * scalar);
    }

    // Dijeljenje sa skalarom
    Vec2 operator/(double scalar) const {
        return Vec2(x / scalar, y / scalar);
    }

    // Compound operatori za direktnu modifikaciju
    Vec2& operator+=(const Vec2& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    Vec2& operator-=(const Vec2& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    Vec2& operator*=(double scalar) {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    Vec2& operator/=(double scalar) {
        x /= scalar;
        y /= scalar;
        return *this;
    }

    // Negativan vektor
    Vec2 operator-() const {
        return Vec2(-x, -y);
    }

    // Magnituda (dužina) vektora
    double mag() const {
        return std::sqrt(x * x + y * y);
    }

    // Kvadrat magnitude (brže za računanje kad ne treba prava dužina)
    double mag_sq() const {
        return x * x + y * y;
    }
};

// Operator za množenje skalara sa vektorom (double * Vec2)
// Ovo omogućava da pišemo: 5.0 * vector
inline Vec2 operator*(double scalar, const Vec2& vec) {
    return Vec2(vec.x * scalar, vec.y * scalar);
}

#endif // VEC2_H