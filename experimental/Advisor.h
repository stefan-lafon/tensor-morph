#ifndef TENSORMORPH_ADVISOR_H
#define TENSORMORPH_ADVISOR_H

#include <vector>
#include <string>

namespace mlir {
namespace tensormorph {

enum AdvisorMode { None, Memory, Compute, Mock };

/**
 * Abstract base class for AI hardware advisors.
 */
class Advisor {
public:
    virtual ~Advisor() = default;
    virtual float Predict(const std::vector<float>& features) const = 0;
    virtual std::string GetProfileName() const = 0;
};

/**
 * Mock advisor for testing the compiler plumbing.
 */
class MockAdvisor : public Advisor {
public:
    explicit MockAdvisor(float fixedScore) : score(fixedScore) {}
    float Predict(const std::vector<float>& features) const override { return score; }
    std::string GetProfileName() const override { return "Mock"; }
private:
    float score;
};

} // namespace tensormorph
} // namespace mlir

#endif // TENSORMORPH_ADVISOR_H