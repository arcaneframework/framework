#ifndef ALIENSTOKESMODULE_H
#define ALIENSTOKESMODULE_H

#include "AlienStokes_axl.h"

#include <arcane/random/Uniform01.h>
#include <arcane/random/LinearCongruential.h>

class MemoryAllocationTracker;

using namespace Arcane;

class AlienStokesModule : public ArcaneAlienStokesObject
{
 public:
  //! Constructor
  AlienStokesModule(const Arcane::ModuleBuildInfo& mbi)
  : ArcaneAlienStokesObject(mbi)
  , m_uniform(m_generator)
  {
  }

  //! Destructor
  virtual ~AlienStokesModule(){};

  static const Integer dim = 3;
  typedef enum { Dirichlet, Neumann } eBCType;

 public:
  //! Initialization
  void init();
  //! Run the test
  void test();
  void initVelocityAndPressure();
  void initSourceAndBoundaryTerm();

  bool solveUzawaMethod(Alien::Matrix& A, Alien::Matrix& B, Alien::Matrix& tB,
      Alien::Vector& f, Alien::Vector& g, Alien::Vector& uk, Alien::Vector& pk);

 private:
  Integer _computeOrientation(Face const& face);
  Integer _computeDir(Real3 const& x);
  bool _isNodeOfFace(Face const& face, Integer node_lid);
  void _computeFaceConnectivity(
      Face const& face, Alien::UniqueArray2<Integer>& connectivity);
  Integer _upStreamFace(Face const& face, Integer cell_id);
  eBCType getBCType(Real3 const& xF, Integer dir);
  eBCType getBCType(Face const& face);

  Real pressure(Real3 const& x) const;
  Real ux(Real3 const& x) const;
  Real uy(Real3 const& x) const;
  Real uz(Real3 const& x) const;
  Real duxdn(Real3 const& x, Integer dir) const;
  Real duydn(Real3 const& x, Integer dir) const;
  Real duzdn(Real3 const& x, Integer dir) const;
  Real div(Real3 const& xC) const;
  Real func(Real3 const& xC, Integer dir) const;
  Real funcN(Real3 const& xF, Integer dir) const;
  Real funcD(Real3 const& xF, Integer dir) const;

  bool m_homogeneous = false;

  Real m_h[dim];
  Real m_h2[dim];

  IParallelMng* m_parallel_mng = nullptr;

  Arcane::CellGroup m_areaU;
  Arcane::random::MinstdRand m_generator;
  mutable Arcane::random::Uniform01<Arcane::random::MinstdRand> m_uniform;

  Alien::MatrixDistribution m_mdist;
  Alien::VectorDistribution m_vdist;

  Alien::MatrixDistribution m_Adist;
  Alien::MatrixDistribution m_Bdist;
  Alien::MatrixDistribution m_tBdist;
  Alien::VectorDistribution m_udist;
  Alien::VectorDistribution m_pdist;

  Real m_omega = 0.5;
  Integer m_uzawa_max_nb_iterations = 0;
};

#endif
