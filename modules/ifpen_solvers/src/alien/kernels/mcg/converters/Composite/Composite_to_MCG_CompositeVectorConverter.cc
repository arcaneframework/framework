#include <alien/core/backend/VectorConverterRegisterer.h>

#include <alien/kernels/composite/data_structure/CompositeVector.h>
#include <alien/kernels/mcg/data_structure/MCGCompositeVector.h>
#include <alien/kernels/mcg/MCGBackEnd.h>
#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/kernels/simple_csr/data_structure/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Composite_to_MCG_CompositeVectorConverter : public Alien::IVectorConverter
{
 public:
  Composite_to_MCG_CompositeVectorConverter() {}

  virtual ~Composite_to_MCG_CompositeVectorConverter() {}

 public:
  Alien::BackEndId sourceBackend() const
  {
    return backendId<Alien::BackEnd::tag::composite>();
  }
  Alien::BackEndId targetBackend() const
  {
    return backendId<Alien::BackEnd::tag::mcgsolver_composite>();
  }

  void convert(
      const Alien::IVectorImpl* sourceImpl, Alien::IVectorImpl* targetImpl) const;

 private:
  void convert(
      const Alien::IVectorImpl* sourceImpl, Alien::IVectorImpl* targetImpl, int i) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
Composite_to_MCG_CompositeVectorConverter::convert(
    const Alien::IVectorImpl* sourceImpl, Alien::IVectorImpl* targetImpl) const
{
  const auto& v = cast<Alien::CompositeKernel::Vector>(sourceImpl, sourceBackend());
  alien_debug([&] {
    cout() << "Converting CompositeVector: " << &v << " to MCGCompositeVector";
  });

  for (Alien::Integer i = 0; i < v.size(); ++i)
    convert(sourceImpl, targetImpl, i);
}

/*---------------------------------------------------------------------------*/

void
Composite_to_MCG_CompositeVectorConverter::convert(
    const Alien::IVectorImpl* sourceImpl, Alien::IVectorImpl* targetImpl, int i) const
{
  const auto& compo = cast<Alien::CompositeKernel::Vector>(sourceImpl, sourceBackend());
  const auto& v = compo[i].impl()->get<Alien::BackEnd::tag::simplecsr>();
  auto& v2 = cast<Alien::MCGCompositeVector>(targetImpl, targetBackend());

  auto values = v.values();
  v2.setValues(i, values.unguardedBasePointer());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(Composite_to_MCG_CompositeVectorConverter);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
