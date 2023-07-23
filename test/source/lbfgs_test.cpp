#include <iomanip>

#include <nano/function/util.h>
#include <nano/solver/lbfgs.h>

#include "lbfgs/solver.hpp"

struct rosenbrock_function : public nano::function_t
{
    using scalar_t = nano::scalar_t;
    using vector_t = Eigen::Matrix<scalar_t, -1, 1>;

    explicit rosenbrock_function(int dim)
        : nano::function_t("rosenbrock", dim)
    {
        convex(nano::convexity::no);
        smooth(nano::smoothness::no);
    }

    auto clone() const -> nano::rfunction_t override { return std::make_unique<rosenbrock_function>(*this); }

    auto operator()(scalar_t const* x, scalar_t* gx, int64_t size) const noexcept {
        auto const n = size;

        scalar_t f {0.0};
        for (auto i = 0L; i < n - 1; ++i) {
            auto const u = x[i];
            auto const v = x[i+1];
            auto const w = u*u-v;

            f += 100 * w*w + (u-1)*(u-1);
        }

        if (gx != nullptr) {
            for (auto i = 0L; i < n - 1; ++i) {
                auto const u = x[i];
                auto const v = x[i+1];

                if (i == 0) {
                    *(gx+i) = 400 * u * (u*u - v) + 2 * (u-1);
                } else {
                    auto const w = x[i-1];
                    *(gx+i) = -200 * (w*w - u) + 400 * (u*u - v) + 2 * (u-1);
                }
            }
            if (n > 1) {
                *(gx+n-1) = -200 * (x[n-2] * x[n-2] - x[n-1]);
            }
        }
        return f;
    }

    auto do_vgrad(nano::vector_t const& x, nano::vector_t* gx) const noexcept -> scalar_t final
    {
        auto* p = gx != nullptr ? gx->data() : nullptr;
        return (*this)(x.data(), p, x.size()); 
    }

    auto operator()(Eigen::Ref<vector_t const> const x) const noexcept {
        return (*this)(x.data(), nullptr, x.size());
    }

    auto operator()(Eigen::Ref<vector_t const> const x, Eigen::Ref<vector_t> gx) const noexcept {
        return (*this)(x.data(), gx.data(), x.size());
    }

    auto operator()(nano::vector_t const& x, nano::vector_t& gx) const noexcept -> scalar_t {
        return do_vgrad(x, &gx);
    }
};

auto solve_libnano(auto function, auto x0) {
    using nano::make_random_vector;
    using nano::scalar_t;
    using nano::solver_lbfgs_t;
    using nano::solver_state_t;
    using nano::vector_t;

    auto solver = nano::solver_lbfgs_t {};
    solver.parameter("solver::lbfgs::history") = 6;
    solver.parameter("solver::epsilon") = 1e-6;
    solver.parameter("solver::max_evals") = 100;
    solver.parameter("solver::tolerance") = std::make_tuple(1e-4, 9e-1);
    solver.lsearch0("constant");
    solver.lsearchk("morethuente");

    const auto state = solver.minimize(function, x0);
    return state.x();
}

auto solve_lbfgs(auto function, auto x0) {
    lbfgs::solver<rosenbrock_function> solver(function);

    if (auto res = solver.optimize(x0)) {
        return res.value();
    }
    return x0; // optimization failed
}

auto main() -> int
{
    auto constexpr dim{3};
    rosenbrock_function function(dim);

    auto n1{0};
    auto n2{0};

    for (auto i = 0; i < 1000; ++i) {
        nano::vector_t x0 = nano::make_random_vector<nano::scalar_t>(dim);

        auto a = solve_libnano(function, x0);
        auto b = solve_lbfgs(function, x0);

        n1 += std::abs(function(a)) < 1e-6;
        n2 += std::abs(function(b)) < 1e-6;
    }

    std::cout << "libnano : minimum found " << n1 << " times\n";
    std::cout << "liblbfgs: minimum found " << n2 << " times\n";

    return 0;
}
