#ifndef BODY_H
#define BODY_H

#include "Vec2.h"

// Klasa koja predstavlja jedno nebesko tijelo u simulaciji
class Body {
public:
    Vec2 pos;   // Pozicija tijela
    Vec2 vel;   // Brzina tijela
    Vec2 acc;   // Akceleracija tijela
    double mass; // Masa tijela

    // Konstruktor za kreiranje tijela sa zadatim parametrima
    Body(const Vec2& pos, const Vec2& vel, double mass)
        : pos(pos), vel(vel), acc(Vec2::zero()), mass(mass) {}

    // Metoda za ažuriranje pozicije i brzine tijela
    // Koristi Verlet integraciju za numeričko rješavanje
    void update(double dt) {
        // Ažuriraj poziciju na osnovu trenutne brzine
        pos += vel * dt;
        
        // Ažuriraj brzinu na osnovu trenutne akceleracije
        vel += acc * dt;
        
        // Resetuj akceleraciju (nova će biti izračunata u sljedećem koraku)
        acc = Vec2::zero();
    }
};

#endif // BODY_H