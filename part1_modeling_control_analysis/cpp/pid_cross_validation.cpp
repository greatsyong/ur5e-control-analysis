#include <array>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

constexpr int N = 6;

using Vec6 = std::array<double, N>;

class JointPIDController
{
public:
    JointPIDController(
        const Vec6& kp,
        const Vec6& ki,
        const Vec6& kd,
        double dt,
        const Vec6& torque_limits)
        : kp_(kp),
          ki_(ki),
          kd_(kd),
          dt_(dt),
          torque_limits_(torque_limits)
    {
        integral_.fill(0.0);
    }

    void reset()
    {
        integral_.fill(0.0);
    }

    Vec6 compute(
    const Vec6& q,
    const Vec6& qdot,
    const Vec6& q_ref,
    const Vec6& qdot_ref)
{
    Vec6 tau{};

    for (int i = 0; i < N; ++i)
    {
        const double error =
            q_ref[i] - q[i];

        const double error_dot =
            qdot_ref[i] - qdot[i];

        // -----------------------------------------------------
        // Torque using CURRENT integral state
        // Matches Python controllers/pid.py
        // -----------------------------------------------------

        const double tau_current =
            kp_[i] * error
            + kd_[i] * error_dot
            + ki_[i] * integral_[i];

        // -----------------------------------------------------
        // Conditional-integration anti-windup
        //
        // Python logic:
        //
        // below_upper = tau_current < limit
        // above_lower = tau_current > -limit
        //
        // integrate_positive =
        //     below_upper || (error < 0)
        //
        // integrate_negative =
        //     above_lower || (error > 0)
        //
        // integrate =
        //     integrate_positive && integrate_negative
        // -----------------------------------------------------

        const double limit =
            torque_limits_[i];

        const bool below_upper =
            tau_current < limit;

        const bool above_lower =
            tau_current > -limit;

        const bool integrate_positive =
            below_upper || (error < 0.0);

        const bool integrate_negative =
            above_lower || (error > 0.0);

        const bool integrate =
            integrate_positive
            && integrate_negative;

        if (integrate)
        {
            integral_[i] +=
                error * dt_;
        }

        // -----------------------------------------------------
        // Final torque using UPDATED integral state
        // -----------------------------------------------------

        const double tau_unsaturated =
            kp_[i] * error
            + kd_[i] * error_dot
            + ki_[i] * integral_[i];

        tau[i] = std::clamp(
            tau_unsaturated,
            -limit,
            limit);
    }

    return tau;
}

    const Vec6& integral() const
    {
        return integral_;
    }

private:
    Vec6 kp_;
    Vec6 ki_;
    Vec6 kd_;

    double dt_;

    Vec6 torque_limits_;
    Vec6 integral_;
};


void print_vector(
    const std::string& name,
    const Vec6& v)
{
    std::cout << name << "\n[";

    for (int i = 0; i < N; ++i)
    {
        std::cout
            << std::scientific
            << std::setprecision(10)
            << v[i];

        if (i < N - 1)
            std::cout << ", ";
    }

    std::cout << "]\n\n";
}


int main()
{
    constexpr double DT = 0.002;
    constexpr double WN = 20.0;
    constexpr double ZETA = 3.0;
    constexpr double ALPHA_I = 18.0;

    // diag(M(q_start))
    const Vec6 effective_inertia = {
        1.05863580,
        2.59135828,
        0.881358542,
        0.0231437242,
        0.00503939441,
        0.00025756
    };

    const Vec6 torque_limits = {
        150.0,
        150.0,
        150.0,
        28.0,
        28.0,
        28.0
    };

    Vec6 kp{};
    Vec6 kd{};
    Vec6 ki{};

    for (int i = 0; i < N; ++i)
    {
        kp[i] =
            effective_inertia[i]
            * WN * WN;

        kd[i] =
            2.0
            * ZETA
            * effective_inertia[i]
            * WN;

        ki[i] =
            ALPHA_I
            * kp[i];
    }

    std::cout
        << "========================================\n"
        << "UR5e PID C++ CROSS-VALIDATION\n"
        << "========================================\n\n";

    std::cout
        << "wn      = " << WN << " rad/s\n"
        << "zeta    = " << ZETA << "\n"
        << "alpha_i = " << ALPHA_I << " 1/s\n"
        << "dt      = " << DT << " s\n\n";

    print_vector("Kp:", kp);
    print_vector("Kd:", kd);
    print_vector("Ki:", ki);


    JointPIDController controller(
        kp,
        ki,
        kd,
        DT,
        torque_limits);


    // ========================================================
    // Test 1
    // Zero tracking error
    // ========================================================

    Vec6 q = {
        0.0,
        -M_PI / 2.0,
        M_PI / 2.0,
        -M_PI / 2.0,
        -M_PI / 2.0,
        0.0
    };

    Vec6 qdot{};
    Vec6 q_ref = q;
    Vec6 qdot_ref{};

    Vec6 tau_zero =
        controller.compute(
            q,
            qdot,
            q_ref,
            qdot_ref);

    print_vector(
        "Test 1 - zero-error torque:",
        tau_zero);


    // ========================================================
    // Test 2
    // +1 degree J2 position error
    // ========================================================

    controller.reset();

    q_ref = q;

    q_ref[1] +=
        M_PI / 180.0;

    Vec6 tau_j2 =
        controller.compute(
            q,
            qdot,
            q_ref,
            qdot_ref);

    print_vector(
        "Test 2 - +1 deg J2 error torque:",
        tau_j2);


    // ========================================================
    // Test 3
    // Mixed position + velocity error
    // ========================================================

    controller.reset();

    q_ref = q;

    q_ref[0] += 0.01;
    q_ref[1] -= 0.02;
    q_ref[2] += 0.015;
    q_ref[3] -= 0.03;
    q_ref[4] += 0.025;
    q_ref[5] -= 0.01;

    qdot = {
        0.10,
        -0.20,
        0.15,
        -0.10,
        0.05,
        -0.08
    };

    qdot_ref = {
        0.15,
        -0.10,
        0.05,
        -0.05,
        0.10,
        -0.02
    };

    Vec6 tau_mixed =
        controller.compute(
            q,
            qdot,
            q_ref,
            qdot_ref);

    print_vector(
        "Test 3 - mixed-error torque:",
        tau_mixed);


    // ========================================================
    // Test 4
    // Anti-windup under repeated saturation
    // ========================================================

    controller.reset();

    q.fill(0.0);
    qdot.fill(0.0);

    q_ref.fill(10.0);
    qdot_ref.fill(0.0);

    Vec6 tau_sat{};

    for (int k = 0; k < 1000; ++k)
    {
        tau_sat =
            controller.compute(
                q,
                qdot,
                q_ref,
                qdot_ref);
    }

    print_vector(
        "Test 4 - saturated torque:",
        tau_sat);

    print_vector(
        "Test 4 - integral state after 1000 saturated steps:",
        controller.integral());


    std::cout
        << "Cross-validation run completed.\n";

    return 0;
}