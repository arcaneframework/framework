// -*- C++ -*-
#ifndef ALIEN_MCGIMPL_MCGCOMPOSITEVECTOR_H
#define ALIEN_MCGIMPL_MCGCOMPOSITEVECTOR_H

#include <ALIEN/Utils/Precomp.h>
#include <ALIEN/Kernels/MCG/MCGPrecomp.h>

#include <ALIEN/Core/Impl/IVectorImpl.h>
#include <ALIEN/Data/ISpace.h>

/*---------------------------------------------------------------------------*/

BEGIN_MCGINTERNAL_NAMESPACE

class CompositeVectorInternal;

END_MCGINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/

BEGIN_NAMESPACE(Alien)

/*---------------------------------------------------------------------------*/

class MCGCompositeVector : public IVectorImpl
{
 public:
  typedef MCGInternal::CompositeVectorInternal VectorInternal;

 public:
  MCGCompositeVector(const MultiVectorImpl* multi_impl);

  virtual ~MCGCompositeVector();

 public:
  void init(const VectorDistribution& dist, const bool need_allocate);
  void allocate();

  void free() {}
  void clear() {}

 public:
  void setValues(const int part, const double* values);
  void getValues(const int part, double* values) const;

 public:
  VectorInternal* internal() { return m_internal; }
  const VectorInternal* internal() const { return m_internal; }

  void update(const MCGCompositeVector& v);

 private:
  bool assemble();

 private:
  VectorInternal* m_internal = nullptr;
};

/*---------------------------------------------------------------------------*/

END_NAMESPACE

/*---------------------------------------------------------------------------*/

#endif /* ALIEN_MCGIMPL_MCGCOMPOSITEVECTOR_H */
