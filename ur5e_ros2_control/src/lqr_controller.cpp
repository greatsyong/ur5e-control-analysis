#include "ur5e_ros2_control/lqr_controller.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace ur5e_ros2_control
{

namespace
{

constexpr double PI = 3.14159265358979323846;

// UR5e physical-link masses
constexpr double MASS[6] = {
  3.761,
  8.058,
  2.846,
  1.370,
  1.300,
  0.365
};

// CoM positions in physical link frames
constexpr double COM[6][3] = {
  { 0.0,    -0.00193, -0.02561  },
  {-0.2125,  0.0,       0.11336  },
  {-0.2422,  0.0,       0.02650  },
  { 0.0,    -0.01634,  -0.00180  },
  { 0.0,     0.01634,  -0.00180  },
  { 0.0,     0.0,      -0.001159 }
};

struct Vec3
{
  double x, y, z;
};

struct Mat3
{
  double a[3][3]{};
};

Vec3 operator+(const Vec3 & a, const Vec3 & b)
{
  return {a.x+b.x, a.y+b.y, a.z+b.z};
}

Vec3 operator-(const Vec3 & a, const Vec3 & b)
{
  return {a.x-b.x, a.y-b.y, a.z-b.z};
}

Vec3 operator*(const Mat3 & R, const Vec3 & v)
{
  return {
    R.a[0][0]*v.x + R.a[0][1]*v.y + R.a[0][2]*v.z,
    R.a[1][0]*v.x + R.a[1][1]*v.y + R.a[1][2]*v.z,
    R.a[2][0]*v.x + R.a[2][1]*v.y + R.a[2][2]*v.z
  };
}

Mat3 multiply(const Mat3 & A, const Mat3 & B)
{
  Mat3 C{};

  for (int i=0;i<3;++i)
    for (int j=0;j<3;++j)
      for (int k=0;k<3;++k)
        C.a[i][j] += A.a[i][k]*B.a[k][j];

  return C;
}

Mat3 rotX(double t)
{
  const double c=std::cos(t);
  const double s=std::sin(t);

  return {{
    {1,0,0},
    {0,c,-s},
    {0,s,c}
  }};
}

Mat3 rotZ(double t)
{
  const double c=std::cos(t);
  const double s=std::sin(t);

  return {{
    {c,-s,0},
    {s,c,0},
    {0,0,1}
  }};
}

Vec3 matVec(const Mat3 & R, const Vec3 & p)
{
  return R*p;
}

Vec3 rotateThenAdd(
  const Mat3 & R,
  const Vec3 & origin,
  const Vec3 & local)
{
  return origin + R*local;
}

Vec3 cross(
  const Vec3 & a,
  const Vec3 & b)
{
  return {
    a.y*b.z-a.z*b.y,
    a.z*b.x-a.x*b.z,
    a.x*b.y-a.y*b.x
  };
}

struct Transform
{
  Mat3 R;
  Vec3 p;
};

Transform multiply(
  const Transform & A,
  const Transform & B)
{
  Transform C;

  C.R = multiply(A.R,B.R);
  C.p = A.p + A.R*B.p;

  return C;
}

Transform identity()
{
  return {
    {{
      {1,0,0},
      {0,1,0},
      {0,0,1}
    }},
    {0,0,0}
  };
}

Transform translation(
  double x,
  double y,
  double z)
{
  return {
    {{
      {1,0,0},
      {0,1,0},
      {0,0,1}
    }},
    {x,y,z}
  };
}

Transform rotation(const Mat3 & R)
{
  return {R,{0,0,0}};
}

}  // namespace


LqrController::LqrController(
  const JointArray & torque_limits,
  double dt)
: torque_limits_(torque_limits),
  dt_(dt)
{
}


LqrController::JointArray
LqrController::gravity(
  const JointArray & q) const
{
  JointArray g{};

  Transform T = identity();

  // Same physical-link transform convention
  // used by the validated UR5e dynamics model.

  T = multiply(
    T,
    rotation(rotZ(PI))
  );

  std::array<Vec3,6> origins{};
  std::array<Vec3,6> axes{};
  std::array<Vec3,6> com_positions{};

  const Vec3 z_axis{0.0,0.0,1.0};


  // ----------------------------------------------------------
  // Joint 1
  // ----------------------------------------------------------

  T = multiply(
    T,
    translation(0.0,0.0,0.1625)
  );

  origins[0] = T.p;

  axes[0] = T.R*z_axis;

  T = multiply(
    T,
    rotation(rotZ(q[0]))
  );

  com_positions[0] =
    T.p + T.R*Vec3{
      COM[0][0],
      COM[0][1],
      COM[0][2]
    };


  // ----------------------------------------------------------
  // Joint 2
  // ----------------------------------------------------------

  T = multiply(
    T,
    rotation(rotX(PI/2.0))
  );

  origins[1] = T.p;
  axes[1] = T.R*z_axis;

  T = multiply(
    T,
    rotation(rotZ(q[1]))
  );

  com_positions[1] =
    T.p + T.R*Vec3{
      COM[1][0],
      COM[1][1],
      COM[1][2]
    };


  // ----------------------------------------------------------
  // Joint 3
  // ----------------------------------------------------------

  T = multiply(
    T,
    translation(-0.425,0.0,0.0)
  );

  origins[2] = T.p;
  axes[2] = T.R*z_axis;

  T = multiply(
    T,
    rotation(rotZ(q[2]))
  );

  com_positions[2] =
    T.p + T.R*Vec3{
      COM[2][0],
      COM[2][1],
      COM[2][2]
    };


  // ----------------------------------------------------------
  // Joint 4
  // ----------------------------------------------------------

  T = multiply(
    T,
    translation(-0.3922,0.0,0.1333)
  );

  origins[3] = T.p;
  axes[3] = T.R*z_axis;

  T = multiply(
    T,
    rotation(rotZ(q[3]))
  );

  com_positions[3] =
    T.p + T.R*Vec3{
      COM[3][0],
      COM[3][1],
      COM[3][2]
    };


  // ----------------------------------------------------------
  // Joint 5
  // ----------------------------------------------------------

  T = multiply(
    T,
    translation(0.0,-0.0997,0.0)
  );

  T = multiply(
    T,
    rotation(rotX(PI/2.0))
  );

  origins[4] = T.p;
  axes[4] = T.R*z_axis;

  T = multiply(
    T,
    rotation(rotZ(q[4]))
  );

  com_positions[4] =
    T.p + T.R*Vec3{
      COM[4][0],
      COM[4][1],
      COM[4][2]
    };


  // ----------------------------------------------------------
  // Joint 6
  // ----------------------------------------------------------

  T = multiply(
    T,
    translation(0.0,0.0996,0.0)
  );

  T = multiply(
    T,
    rotation(rotX(-PI/2.0))
  );

  origins[5] = T.p;
  axes[5] = T.R*z_axis;

  T = multiply(
    T,
    rotation(rotZ(q[5]))
  );

  com_positions[5] =
    T.p + T.R*Vec3{
      COM[5][0],
      COM[5][1],
      COM[5][2]
    };


  // ----------------------------------------------------------
  // Gravity generalized force
  //
  // g(q) = sum Jv_i^T m_i g
  //
  // Gravity = [0,0,-9.81] m/s²
  // ----------------------------------------------------------

  for (std::size_t i=0;i<N;++i)
  {
    for (std::size_t j=0;j<=i;++j)
    {
      const Vec3 r =
        com_positions[i] - origins[j];

      const Vec3 Jv =
        cross(
          axes[j],
          r
        );

      g[j] +=
        MASS[i]
        * (9.81)
        * Jv.z;
    }
  }

  return g;
}


LqrController::JointArray
LqrController::compute(
  const JointArray & q,
  const JointArray & qdot,
  const JointArray & q_ref,
  const JointArray & qdot_ref)
{
  StateArray dx{};

  for (std::size_t i=0;i<N;++i)
  {
    dx[i] =
      q[i] - q_ref[i];

    dx[i+N] =
      qdot[i] - qdot_ref[i];
  }


  // ----------------------------------------------------------
  // IMPORTANT:
  //
  // Final Part 1B LQR:
  //
  // tau = g(q_ref) - K * dx
  //
  // NOT g(q_start)
  // NOT g(q)
  // NOT computed torque
  // ----------------------------------------------------------

  const JointArray tau_ff =
    gravity(q_ref);


  JointArray tau{};

  for (std::size_t i=0;i<N;++i)
  {
    double Kdx = 0.0;

    for (std::size_t j=0;j<NX;++j)
    {
      Kdx +=
        K[i][j] * dx[j];
    }

    const double tau_unsaturated =
      tau_ff[i] - Kdx;

    tau[i] =
      std::clamp(
        tau_unsaturated,
        -torque_limits_[i],
         torque_limits_[i]
      );
  }

  return tau;
}


void LqrController::reset()
{
  // No integral or internal dynamic state.
}

}  // namespace ur5e_ros2_control