// %ignore must be before %include of non-template declaration
%ignore Arcane::Hdf5Utils::_ArcaneHdf5UtilsGroupIterateMe;

%include arcane/std/Hdf5Utils.h

%define SWIG_ARCANE_HDF5UTILS_STANDARD_SCALAR_SPECIALIZE( DATATYPE )
  typedef Arcane::Hdf5Utils::StandardScalarT<DATATYPE> Hdf5Utils##DATATYPE##Scalar;
  %template(Hdf5Utils##DATATYPE##Scalar) Arcane::Hdf5Utils::StandardScalarT<DATATYPE>;
%enddef

%define SWIG_ARCANE_HDF5UTILS_STANDARD_ARRAY_SPECIALIZE( DATATYPE )
  // %ignore and extensions must be after %include of template declaration but before %template instanciation
  %ignore Arcane::Hdf5Utils::StandardArrayT<DATATYPE>::directRead;

  %typemap(cscode) Arcane::Hdf5Utils::StandardArrayT<DATATYPE>
  %{
      public void DirectRead(StandardTypes types, Arcane.##DATATYPE##Array buffer) 
      {
        ReadDim();
				buffer.Resize (Dimensions()[0]);
				Read (types, buffer.View);
      }
  %}

  typedef Arcane::Hdf5Utils::StandardArrayT<DATATYPE> Hdf5Utils##DATATYPE##Array;
  %template(Hdf5Utils##DATATYPE##Array) Arcane::Hdf5Utils::StandardArrayT<DATATYPE>;
%enddef


SWIG_ARCANE_HDF5UTILS_STANDARD_SCALAR_SPECIALIZE(Byte)
SWIG_ARCANE_HDF5UTILS_STANDARD_SCALAR_SPECIALIZE(Int16)
SWIG_ARCANE_HDF5UTILS_STANDARD_SCALAR_SPECIALIZE(Int32)
SWIG_ARCANE_HDF5UTILS_STANDARD_SCALAR_SPECIALIZE(Int64)
SWIG_ARCANE_HDF5UTILS_STANDARD_SCALAR_SPECIALIZE(Real)
SWIG_ARCANE_HDF5UTILS_STANDARD_SCALAR_SPECIALIZE(String)
SWIG_ARCANE_HDF5UTILS_STANDARD_SCALAR_SPECIALIZE(Real2)
SWIG_ARCANE_HDF5UTILS_STANDARD_SCALAR_SPECIALIZE(Real3)

// NOTE: IFPEN signale un problème de compilation pour le type Byte.
 // TODO: vérifier si c'est toujours le cas. Du côté CEA, il n'y
// a pas de problèmes.
SWIG_ARCANE_HDF5UTILS_STANDARD_ARRAY_SPECIALIZE(Byte)
SWIG_ARCANE_HDF5UTILS_STANDARD_ARRAY_SPECIALIZE(Int16)
SWIG_ARCANE_HDF5UTILS_STANDARD_ARRAY_SPECIALIZE(Int32)
SWIG_ARCANE_HDF5UTILS_STANDARD_ARRAY_SPECIALIZE(Int64)
SWIG_ARCANE_HDF5UTILS_STANDARD_ARRAY_SPECIALIZE(Real)
SWIG_ARCANE_HDF5UTILS_STANDARD_ARRAY_SPECIALIZE(Real2)
SWIG_ARCANE_HDF5UTILS_STANDARD_ARRAY_SPECIALIZE(Real3)


