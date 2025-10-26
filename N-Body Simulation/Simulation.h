#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "Body.h"

// Konstante za simulaciju
const double TAU = 2.0 * M_PI;  // 2π
const double DT = 0.000001;     // Vremenski korak simulacije (jako mali za preciznost)
const double MIN = 0.0001;      // Minimalna udaljenost da izbjegnemo singularitet

class Simulation {
private:
    std::mt19937_64 rng; // Random number generator

    // Generira nasumičnu tačku na jediničnom disku
    Vec2 rand_disc() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double theta = dist(rng) * TAU;  // Nasumični ugao
        double r = dist(rng);             // Nasumična udaljenost od centra
        return Vec2(std::cos(theta), std::sin(theta)) * r;
    }

    // Kreira nasumično tijelo sa pozicijom, brzinom i masom
    Body rand_body() {
        Vec2 pos = rand_disc();
        Vec2 vel = rand_disc();
        return Body(pos, vel, 1.0); // Sva tijela imaju masu 1.0
    }

public:
    std::vector<Body> bodies; // Lista svih tijela u simulaciji

    // Konstruktor - inicijalizuje simulaciju sa zadatim seed-om
    Simulation(uint64_t seed) : rng(seed) {
        const int n = 3; // Broj tijela u simulaciji
        
        // Generiši n nasumičnih tijela
        for (int i = 0; i < n; i++) {
            bodies.push_back(rand_body());
        }

        // Izračunaj srednju brzinu sistema (da bi sistem bio u centru mase)
        Vec2 vel_sum = Vec2::zero();
        Vec2 pos_sum = Vec2::zero();
        
        for (const auto& b : bodies) {
            vel_sum += b.vel * b.mass;
            pos_sum += b.pos * b.mass;
        }
        
        Vec2 avg_vel = vel_sum / static_cast<double>(n);
        Vec2 avg_pos = pos_sum / static_cast<double>(n);

        // Centriranje sistema: oduzmi srednju brzinu i poziciju
        for (auto& b : bodies) {
            b.vel -= avg_vel;
            b.pos -= avg_pos;
        }

        // Normalizuj pozicije tako da najdalje tijelo bude na jediničnoj udaljenosti
        double max_r = 0.0;
        for (const auto& b : bodies) {
            max_r = std::max(max_r, b.pos.mag());
        }
        
        for (auto& b : bodies) {
            b.pos /= max_r;
        }
    }

    // Glavna funkcija simulacije - računa gravitacione sile i ažurira tijela
    void update() {
        // Izračunaj gravitacione interakcije između svih parova tijela
        for (size_t i = 0; i < bodies.size(); i++) {
            Vec2 p1 = bodies[i].pos;
            double m1 = bodies[i].mass;
            
            // Razmotri samo parove (i, j) gdje je j > i da izbjegnemo duple kalkulacije
            for (size_t j = i + 1; j < bodies.size(); j++) {
                Vec2 p2 = bodies[j].pos;
                double m2 = bodies[j].mass;
                
                // Vektor od tijela i do tijela j
                Vec2 r = p2 - p1;
                
                // Kvadrat udaljenosti
                double mag_sq = r.mag_sq();
                
                // Udaljenost
                double mag = std::sqrt(mag_sq);
                
                // Gravitaciona sila: F = G * m1 * m2 / r^2
                // Ovdje je G = 1 za jednostavnost
                // tmp = r / (r^3) = smjer * 1/r^2
                Vec2 tmp = r / (std::max(mag_sq, MIN) * mag);
                
                // Primjeni silu na oba tijela (Newtonov treći zakon)
                bodies[i].acc += m2 * tmp;  // Tijelo i privlači tijelo j
                bodies[j].acc -= m1 * tmp;  // Tijelo j privlači tijelo i (suprotni smjer)
            }
        }

        // Ažuriraj pozicije i brzine svih tijela
        for (auto& body : bodies) {
            body.update(DT);
        }
    }
};

#endif // SIMULATION_H