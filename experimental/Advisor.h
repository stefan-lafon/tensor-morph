#pragma once
#include <vector>
#include <string>
#include <memory>

/**
 * Interface for hardware-specific optimization advice.
 * Allows the compiler to stay agnostic of the underlying model type.
 */
class Advisor {
public:
    virtual ~Advisor() = default;

    // Returns a predicted profit ratio for the given tensor shape.
    virtual float Predict(const std::vector<float>& features) const = 0;

    // Identifies the hardware target this model was trained for.
    virtual std::string GetProfileName() const = 0;
};

// Factory function to create the appropriate advisor implementation.
std::unique_ptr<Advisor> CreateAdvisor(const std::string& profile);