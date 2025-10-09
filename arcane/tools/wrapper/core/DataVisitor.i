// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Wrapper pour les différentes classes implémentant IDataVisitor
//
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%feature("director") Arcane::IDataVisitor;
%feature("director") Arcane::AbstractDataVisitor;

%include arcane/core/ISerializedData.h
%include arcane/core/IData.h
%include arcane/core/IDataVisitor.h

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// La vraie classe AbstractDataVisitor utilise un héritage multiple qui
// n'est pas supporté par le C#. Au lieu d'inclure le .h correspondant,
// on définit cette classe ici.
namespace Arcane
{
  class AbstractDataVisitor
  : public IDataVisitor
  {
  public:

    virtual void applyDataVisitor(IScalarData* data);
    virtual void applyDataVisitor(IArrayData* data);
    virtual void applyDataVisitor(IArray2Data* data);
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define SWIG_ARCANE_DATAVISITOR(DATATYPE)
%template(I##DATATYPE##Array2Data) Arcane::IArray2DataT<DATATYPE>;
%template(I##DATATYPE##ArrayData) Arcane::IArrayDataT<DATATYPE>;
%template(I##DATATYPE##ScalarData) Arcane::IScalarDataT<DATATYPE>;
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if !defined(SWIGPYTHON)
SWIG_ARCANE_DATAVISITOR(Byte)
SWIG_ARCANE_DATAVISITOR(Real)
SWIG_ARCANE_DATAVISITOR(Int16)
SWIG_ARCANE_DATAVISITOR(Int32)
SWIG_ARCANE_DATAVISITOR(Int64)
SWIG_ARCANE_DATAVISITOR(Real2)
SWIG_ARCANE_DATAVISITOR(Real3)
SWIG_ARCANE_DATAVISITOR(Real2x2)
SWIG_ARCANE_DATAVISITOR(Real3x3)
#endif

%template(IData_Ref) Arcane::Ref<Arcane::IData>;
%template(ISerializedData_Ref) Arcane::Ref<Arcane::ISerializedData>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
