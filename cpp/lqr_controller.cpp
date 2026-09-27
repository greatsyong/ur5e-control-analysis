#include <array>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

constexpr int NQ = 6;
constexpr int NX = 12;

using Vec6 = std::array<double, NQ>;
using Vec12 = std::array<double, NX>;


// ============================================================
// Final discrete LQR gain
//
// Design:
//   Ts = 0.002 s
//   rho_u = 100
//   Bryson-normalized Q/R
//   fixed linearization at q_start
// ============================================================

const double K[NQ][NX] = {
    {
        3.87331238999120501e+02,
        -7.22587049855960828e+00,
        2.94271819926002376e+00,
        -2.42920521757010111e+00,
        4.24357152976319529e+00,
        -1.12407746759820487e-01,
        8.25740345786492611e+01,
        -3.29118462090232056e+00,
        6.61857112922873636e-01,
        -4.45998322853161877e-01,
        9.82228490386837461e-01,
        -2.37285690084990293e-02
    },
    {
        -9.89182576379690559e+00,
        4.48195612326768185e+02,
        1.08819110134790353e+01,
        -4.21401476409867026e+00,
        1.05642963957153979e+00,
        -2.36727791911153888e-02,
        -3.82563361405613911e+00,
        9.41755086501014205e+01,
        6.32416722637777795e+00,
        -4.10417810564487895e-01,
        2.11834311353356564e-01,
        -4.60909509404735466e-03
    },
    {
        8.70829150371327465e+00,
        3.14519707898608303e+01,
        3.22190153227432575e+02,
        -1.50732348270310883e+01,
        -6.64661699419755392e-01,
        1.35703457351065322e-02,
        1.81063841274797177e+00,
        1.04586537605586205e+01,
        6.87228424010004915e+01,
        -2.46180454796099779e+00,
        -1.33086351096623506e-01,
        2.72209037701236353e-03
    },
    {
        1.87802719152631892e+00,
        -1.15138599130168573e+00,
        3.46238805081915118e+01,
        2.52862167051943914e+01,
        -1.97573907122718995e-01,
        2.94566863916215053e-03,
        4.15817424616144626e-01,
        2.15880880789494051e-01,
        7.49082050753050055e+00,
        5.27960539749384328e+00,
        -3.95712468167413178e-02,
        5.89666989661058265e-04
    },
    {
        1.08442358210963068e+01,
        2.18704878198281039e+00,
        -1.90689862037346458e+00,
        -1.19200694752309544e-01,
        9.89700557966810557e+00,
        1.70907897178721874e-02,
        2.30452318098066433e+00,
        4.39809231868882211e-01,
        -3.80486849219308476e-01,
        -2.38017501433715195e-02,
        2.11583223353523309e+00,
        3.42595349098316756e-03
    },
    {
        -1.02252119026277274e-01,
        -2.06496577866192577e-02,
        1.80642363886902579e-02,
        1.45809354164731027e-03,
        1.92841370459910857e-02,
        6.40516412522741607e-01,
        -2.16931941217040833e-02,
        -4.15160924645441125e-03,
        3.60496933082277106e-03,
        2.91870159164000673e-04,
        3.86284191248474022e-03,
        1.29384661860637407e-01
    }
};


const Vec6 TORQUE_LIMITS = {
    150.0,
    150.0,
    150.0,
    28.0,
    28.0,
    28.0
};


// ============================================================
// Result structure
// ============================================================

struct LQRResult
{
    Vec6 feedback;
    Vec6 unsaturated;
    Vec6 torque;
};


// ============================================================
// Final online LQR control law
// ============================================================

LQRResult compute_lqr(
    const Vec6& q,
    const Vec6& qdot,
    const Vec6& q_ref,
    const Vec6& qdot_ref,
    const Vec6& tau_ff)
{
    Vec12 dx{};

    // Python convention:
    //
    // dx =
    // [
    //   q - q_ref,
    //   qdot - qdot_ref
    // ]

    for (int i = 0; i < NQ; ++i)
    {
        dx[i] =
            q[i] - q_ref[i];

        dx[i + NQ] =
            qdot[i] - qdot_ref[i];
    }


    LQRResult result{};


    for (int i = 0; i < NQ; ++i)
    {
        double Kdx = 0.0;

        for (int j = 0; j < NX; ++j)
        {
            Kdx +=
                K[i][j] * dx[j];
        }

        result.feedback[i] =
            -Kdx;

        result.unsaturated[i] =
            tau_ff[i]
            + result.feedback[i];

        result.torque[i] =
            std::clamp(
                result.unsaturated[i],
                -TORQUE_LIMITS[i],
                TORQUE_LIMITS[i]
            );
    }


    return result;
}


// ============================================================
// Print helper
// ============================================================

void print_vec(
    const char* name,
    const Vec6& v)
{
    std::cout
        << name
        << " = [";

    for (int i = 0; i < NQ; ++i)
    {
        std::cout
            << std::setprecision(15)
            << v[i];

        if (i < NQ - 1)
        {
            std::cout << ", ";
        }
    }

    std::cout << "]\n";
}


// ============================================================
// Deterministic implementation tests
//
// Test vectors will be replaced / compared with the Python
// reference values generated from the nominal model.
// ============================================================

int main()
{
    constexpr double DEG =
        3.14159265358979323846 / 180.0;


    // --------------------------------------------------------
    // Test 1:
    // Zero tracking error.
    //
    // Expected:
    // feedback = 0
    // torque = tau_ff
    // --------------------------------------------------------

    Vec6 q_ref_1 = {
        0.0,
        -90.0 * DEG,
        90.0 * DEG,
        -90.0 * DEG,
        -90.0 * DEG,
        0.0
    };

    Vec6 q_1 = q_ref_1;

    Vec6 qdot_ref_1 = {
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    };

    Vec6 qdot_1 =
        qdot_ref_1;


    // Initial equilibrium gravity from the validated model.
    //
    // tau_eq =
    // [0,
    //  -20.2642668,
    //  -20.2642668,
    //  -1.8251142,
    //   ~0,
    //   0]
    //
    // These values are only for Test 1.
    // Higher precision Python values will be used in the
    // cross-validation script.

    Vec6 tau_ff_1 = {
        0.000000000000000e+00,
        -2.026426684500000e+01,
        -2.026426684500000e+01,
        -1.825114203000000e+00,
        -1.405612856657376e-18,
        0.000000000000000e+00
    };


    auto result_1 = compute_lqr(
        q_1,
        qdot_1,
        q_ref_1,
        qdot_ref_1,
        tau_ff_1
    );


    std::cout
        << "\n========================================\n"
        << "TEST 1: ZERO TRACKING ERROR\n"
        << "========================================\n";

    print_vec(
        "feedback",
        result_1.feedback
    );

    print_vec(
        "unsaturated",
        result_1.unsaturated
    );

    print_vec(
        "torque",
        result_1.torque
    );


    // --------------------------------------------------------
    // Test 2:
    // +1 degree J2 position error
    //
    // q - q_ref = +1 deg on J2.
    // --------------------------------------------------------

    Vec6 q_2 = q_ref_1;

    q_2[1] +=
        1.0 * DEG;


    auto result_2 = compute_lqr(
        q_2,
        qdot_1,
        q_ref_1,
        qdot_ref_1,
        tau_ff_1
    );


    std::cout
        << "\n========================================\n"
        << "TEST 2: +1 DEG J2 POSITION ERROR\n"
        << "========================================\n";

    print_vec(
        "feedback",
        result_2.feedback
    );

    print_vec(
        "unsaturated",
        result_2.unsaturated
    );

    print_vec(
        "torque",
        result_2.torque
    );


    // --------------------------------------------------------
    // Test 3:
    // Mixed position + velocity errors
    // --------------------------------------------------------

    Vec6 q_3 = q_ref_1;

    q_3[0] +=  0.5 * DEG;
    q_3[1] += -0.8 * DEG;
    q_3[2] +=  0.3 * DEG;
    q_3[3] += -1.0 * DEG;
    q_3[4] +=  0.7 * DEG;
    q_3[5] += -0.4 * DEG;


    Vec6 qdot_3 = {
         2.0 * DEG,
        -3.0 * DEG,
         1.5 * DEG,
        -2.5 * DEG,
         1.0 * DEG,
        -1.5 * DEG
    };


    auto result_3 = compute_lqr(
        q_3,
        qdot_3,
        q_ref_1,
        qdot_ref_1,
        tau_ff_1
    );


    std::cout
        << "\n========================================\n"
        << "TEST 3: MIXED POSITION/VELOCITY ERROR\n"
        << "========================================\n";

    print_vec(
        "feedback",
        result_3.feedback
    );

    print_vec(
        "unsaturated",
        result_3.unsaturated
    );

    print_vec(
        "torque",
        result_3.torque
    );


    // --------------------------------------------------------
    // Test 4:
    // Large errors to exercise saturation.
    // --------------------------------------------------------

    Vec6 q_4 = q_ref_1;

    for (int i = 0; i < NQ; ++i)
    {
        q_4[i] +=
            90.0 * DEG;
    }


    Vec6 qdot_4 = {
        100.0 * DEG,
        100.0 * DEG,
        100.0 * DEG,
        100.0 * DEG,
        100.0 * DEG,
        100.0 * DEG
    };


    auto result_4 = compute_lqr(
        q_4,
        qdot_4,
        q_ref_1,
        qdot_ref_1,
        tau_ff_1
    );


    std::cout
        << "\n========================================\n"
        << "TEST 4: SATURATION\n"
        << "========================================\n";

    print_vec(
        "feedback",
        result_4.feedback
    );

    print_vec(
        "unsaturated",
        result_4.unsaturated
    );

    print_vec(
        "torque",
        result_4.torque
    );


    return 0;
}