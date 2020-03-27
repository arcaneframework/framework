#include <alien/core/backend/VectorConverterRegisterer.h>

#include <alien/Kernels/Composite/DataStructure/CompositeVector.h>
#include <alien/Kernels/MCG/DataStructure/MCGVector.h>
#include <alien/Kernels/MCG/MCGBackEnd.h>
#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/kernels/simple_csr/data_structure/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Composite_to_MCG_VectorConverter
  : public Alien::IVectorConverter
{
public:

  Composite_to_MCG_VectorConverter(){}

  virtual ~Composite_to_MCG_VectorConverter() {}

public:

  Alien::BackEndId sourceBackend() const { return backendId<Alien::BackEnd::tag::composite>(); }
  Alien::BackEndId targetBackend() const { return backendId<Alien::BackEnd::tag::mcgsolver>(); }

  void convert(const Alien::IVectorImpl * sourceImpl, Alien::IVectorImpl * targetImpl) const;

private:

  void convert(const Alien::IVectorImpl * sourceImpl, Alien::IVectorImpl * targetImpl, int i) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
Composite_to_MCG_VectorConverter::
convert(const Alien::IVectorImpl * sourceImpl, Alien::IVectorImpl * targetImpl) const
{
  const auto& v = cast<Alien::CompositeKernel::Vector>(sourceImpl, sourceBackend());
  alien_debug([&] {
      cout() << "Converting CompositeVector: " << &v << " to MCGVector";
  });

  for(Alien::Integer i=0;i<v.size();++i)
    convert(sourceImpl,targetImpl,i);
}

/*---------------------------------------------------------------------------*/

void
Composite_to_MCG_VectorConverter::
convert(const Alien::IVectorImpl * sourceImpl, Alien::IVectorImpl * targetImpl, int i) const
{
  const auto& compo = cast<Alien::CompositeKernel::Vector>(sourceImpl, sourceBackend());
  const auto& v = compo[i].impl()->get<Alien::BackEnd::tag::simplecsr>();
  auto& v2 = cast<Alien::MCGVector>(targetImpl, targetBackend());

  auto values = v.values();
  if(i==0)
    v2.setValues(values.size(), Alien::dataPtr(values));
  else
    v2.setExtraEqValues(values.size(), Alien::dataPtr(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

REGISTER_VECTOR_CONVERTER(Composite_to_MCG_VectorConverter);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
