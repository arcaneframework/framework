// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Wrapper pour les classes tableau.
// Definitions pour wrapper les classes:
// - ArrayView
// - ConstArrayView
// - Array2View

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// On ne peut pas inclure 'arcane/utils/UtilsTypes.h' car SWIG (3.0.12) ne
// semble pas bien gérer 'using' des classes de Arccore.
namespace Arccore
{
  template<typename ArgType> class IFunctorWithArgumentT;
}

namespace Arcane
{
  template<class T> class ArrayView;
  template<class T> class ConstArrayView;
  template<typename T> class Array;
  template<typename T> class UniqueArray;
  template<typename T> class SharedArray;
  template<typename T> class Array2;
  template<typename DataType> class UniqueArray2;
  template<typename DataType> class SharedArray2;
  template<typename T> class Array2View;
  template<typename T> class ConstArray2View;
  template<typename T> class MultiArray2View;
  template<typename T> class ConstMultiArray2View;
  template<typename T> class MultiArray2;
  template<typename DataType> class UniqueMultiArray2;
  template<typename DataType> class SharedMultiArray2;

  template<typename T> class EnumeratorT;
  template<typename T> class ListEnumeratorT;

  template<typename T> class Collection;
  template<typename T> class List;

  template<typename... Args> class EventObservable;
  template<typename... Args> class EventObserver;

  typedef Collection<String> StringCollection;

  typedef List<String> StringList;

  typedef ArrayView<Pointer> PointerArrayView;
  typedef ArrayView<UChar> UCharArrayView;
  typedef ArrayView<Integer> IntegerArrayView;
  typedef ArrayView<Real> RealArrayView;
  typedef ArrayView<bool> BoolArrayView;
  typedef ArrayView<String> StringArrayView;

  typedef ConstArrayView<Pointer> PointerConstArrayView;
  typedef ConstArrayView<UChar> UCharConstArrayView;
  typedef ConstArrayView<Integer> IntegerConstArrayView;
  typedef ConstArrayView<Real> RealConstArrayView;
  typedef ConstArrayView<bool> BoolConstArrayView;
  typedef ConstArrayView<String> StringConstArrayView;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef unsigned char Byte;
typedef signed char SByte;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Spécialisation du type 'ConstArrayView'
// - 'CTYPE' est le type C++ à spécialiser
// - 'CSHARP_NAME' est le nom de classe C# correspondant. Le code correspondant
//   doit être fourni par le développeur (i.e: il n'est pas généré par SWIG)
%define ARCANE_SWIG_SPECIALIZE_CONSTARRAYVIEW(CTYPE,CSHARP_NAME)

namespace Arcane
{
  typedef ConstArrayView<CTYPE> CTYPE##ConstArrayView;
}
%typemap(cstype) Arcane::ConstArrayView<CTYPE> %{ CSHARP_NAME %}
%typemap(ctype, out="Arcane::ConstArrayViewPOD",
	 directorout="Arcane::ConstArrayViewPOD",
	 directorin="Arcane::ConstArrayViewPOD") Arcane::ConstArrayView<CTYPE > "Arcane::ConstArrayViewPOD_T<CTYPE >"
%typemap(imtype) Arcane::ConstArrayView<CTYPE> %{ CSHARP_NAME %}
%typemap(csin) Arcane::ConstArrayView<CTYPE> "$csinput"
%typemap(csout) Arcane::ConstArrayView<CTYPE > { return $imcall;  }
%typemap(in) Arcane::ConstArrayView<CTYPE > %{$1 = Arcane::ConstArrayView<CTYPE >($input.m_size,$input.m_ptr); %}
%typemap(directorin) Arcane::ConstArrayView<CTYPE >
%{
   Arcane::ConstArrayView<CTYPE> result_ref_$input = $1;
   $input .m_size = result_ref_$input.size();
   $input .m_ptr = result_ref_$input.data();
%}
%typemap(out) Arcane::ConstArrayView<CTYPE >
%{
   Arcane::ConstArrayView<CTYPE> result_ref = $1;
   $result .m_size = result_ref.size();
   $result .m_ptr = result_ref.data();
%}
%typemap(directorout) Arcane::ConstArrayView<CTYPE >
%{
  $1 = Arcane::ConstArrayView<CTYPE >($input.m_size,$input.m_ptr);
%}
%typemap(csdirectorin) Arcane::ConstArrayView<CTYPE > "$iminput"
%typemap(csdirectorout) Arcane::ConstArrayView<CTYPE > "$cscall"

namespace Arcane
{
  template<> class ConstArrayView<CTYPE > {
    ConstArrayView() {}
  };
}

%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Spécialisation du type 'ArrayView'
// (void doc ARCANE_SWIG_SPECIALIZE_CONSTARRAYVIEW)
%define ARCANE_SWIG_SPECIALIZE_ARRAYVIEW(CTYPE,CSHARP_NAME)

namespace Arcane
{
  typedef ArrayView<CTYPE> CTYPE##ArrayView;
}
%typemap(cstype) Arcane::ArrayView<CTYPE> %{ CSHARP_NAME %}

%typemap(ctype, out="Arcane::ArrayViewPOD",
	 directorout="Arcane::ArrayViewPOD",
	 directorin="Arcane::ArrayViewPOD") Arcane::ArrayView<CTYPE > "Arcane::ArrayViewPOD_T<CTYPE >"

%typemap(imtype) Arcane::ArrayView<CTYPE> %{ CSHARP_NAME %}

%typemap(csin) Arcane::ArrayView<CTYPE> "$csinput"

%typemap(csout) Arcane::ArrayView<CTYPE > { return $imcall;  }

%typemap(in) Arcane::ArrayView<CTYPE > %{$1 = Arcane::ArrayView<CTYPE >($input.m_size,$input.m_ptr); %}
%typemap(directorin) Arcane::ArrayView<CTYPE >
%{
   Arcane::ArrayView<CTYPE> result_ref_$input = $1;
   $input .m_size = result_ref_$input.size();
   $input .m_ptr = result_ref_$input.data();
%}
%typemap(out) Arcane::ArrayView<CTYPE >
%{
   Arcane::ArrayView<CTYPE> result_ref = $1;
   $result .m_size = result_ref.size();
   $result .m_ptr = result_ref.data();
%}
%typemap(directorout) Arcane::ArrayView<CTYPE >
%{
   $1 = Arcane::ArrayView<CTYPE >($input.m_size,$input.m_ptr);
%}
%typemap(csdirectorin) Arcane::ArrayView<CTYPE > "$iminput"
%typemap(csdirectorout) Arcane::ArrayView<CTYPE > "$cscall"

namespace Arcane
{
  template<> class ArrayView<CTYPE > {
    ArrayView() {}
  };
}

%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Spécialise les classes 'ArrayView' et 'ConstArrayView' pour le type
 * C++ 'CTYPE'. Les types C# auront pour nom 'Arcane.CNAME##ArrayView'
 * et 'Arcane.CNAME##ConstArrayView'. Les types C# ne sont pas générés
 * par swig et donc doivent être fournis par le développeur.
 */
%define SWIG_ARCANE_ARRAYVIEW_SPECIALIZE_NEW2(CTYPE,CNAME)
ARCANE_SWIG_SPECIALIZE_ARRAYVIEW(CTYPE,Arcane.CNAME##ArrayView)
ARCANE_SWIG_SPECIALIZE_CONSTARRAYVIEW(CTYPE,Arcane.CNAME##ConstArrayView)
%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Ancienne méthode pour référence. Ne pas utiliser.
%define SWIG_ARCANE_ARRAYVIEW_SPECIALIZE_NEW2_DEPRECATED(CTYPE,CNAME)
namespace Arcane
{
  typedef ArrayView<CTYPE> CTYPE##ArrayView;
  typedef ConstArrayView<CTYPE> CTYPE##ConstArrayView;
}
%typemap(cstype) Arcane::ConstArrayView<CTYPE> %{Arcane.CNAME##ConstArrayView%}
%typemap(cstype) Arcane::ArrayView<CTYPE> %{Arcane.CNAME##ArrayView%}

%typemap(ctype, out="Arcane::ConstArrayViewPOD",
	 directorout="Arcane::ConstArrayViewPOD",
	 directorin="Arcane::ConstArrayViewPOD") Arcane::ConstArrayView<CTYPE > "Arcane::ConstArrayViewPOD_T<CTYPE >"
%typemap(ctype, out="Arcane::ArrayViewPOD",
	 directorout="Arcane::ArrayViewPOD",
	 directorin="Arcane::ArrayViewPOD") Arcane::ArrayView<CTYPE > "Arcane::ArrayViewPOD_T<CTYPE >"

%typemap(imtype) Arcane::ConstArrayView<CTYPE> %{Arcane.CNAME##ConstArrayView%}
%typemap(imtype) Arcane::ArrayView<CTYPE> %{Arcane.CNAME##ArrayView%}

%typemap(csin) Arcane::ConstArrayView<CTYPE> "$csinput"
%typemap(csin) Arcane::ArrayView<CTYPE> "$csinput"

%typemap(csout) Arcane::ConstArrayView<CTYPE > { return $imcall;  }
%typemap(csout) Arcane::ArrayView<CTYPE > { return $imcall;  }

%typemap(in) Arcane::ConstArrayView<CTYPE > %{$1 = Arcane::ConstArrayView<CTYPE >($input.m_size,$input.m_ptr); %}
%typemap(in) Arcane::ArrayView<CTYPE > %{$1 = Arcane::ArrayView<CTYPE >($input.m_size,$input.m_ptr); %}
%typemap(directorin) Arcane::ConstArrayView<CTYPE >
%{
   Arcane::ConstArrayView<CTYPE> result_ref_$input = $1;
   $input .m_size = result_ref_$input.size();
   $input .m_ptr = result_ref_$input.data();
%}
%typemap(directorin) Arcane::ArrayView<CTYPE >
%{
   Arcane::ArrayView<CTYPE> result_ref_$input = $1;
   $input .m_size = result_ref_$input.size();
   $input .m_ptr = result_ref_$input.data();
%}

%typemap(out) Arcane::ConstArrayView<CTYPE >
%{
   Arcane::ConstArrayView<CTYPE> result_ref = $1;
   $result .m_size = result_ref.size();
   $result .m_ptr = result_ref.data();
%}
%typemap(directorout) Arcane::ConstArrayView<CTYPE >
%{
  $1 = Arcane::ConstArrayView<CTYPE >($input.m_size,$input.m_ptr);
%}
%typemap(out) Arcane::ArrayView<CTYPE >
%{
   Arcane::ArrayView<CTYPE> result_ref = $1;
   $result .m_size = result_ref.size();
   $result .m_ptr = result_ref.data();
%}
%typemap(directorout) Arcane::ArrayView<CTYPE >
%{
   $1 = Arcane::ArrayView<CTYPE >($input.m_size,$input.m_ptr);
%}

%typemap(csdirectorin) Arcane::ConstArrayView<CTYPE > "$iminput"
%typemap(csdirectorout) Arcane::ConstArrayView<CTYPE > "$cscall"

%typemap(csdirectorin) Arcane::ArrayView<CTYPE > "$iminput"
%typemap(csdirectorout) Arcane::ArrayView<CTYPE > "$cscall"

namespace Arcane
{
  template<> class ConstArrayView<CTYPE > {
    ConstArrayView() {}
  };
  template<> class ArrayView<CTYPE > {
    ArrayView() {}
  };
}

%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%define SWIG_ARCANE_ARRAY2VIEW_SPECIALIZE_NEW2(CTYPE,CNAME)

%typemap(cstype) Arcane::Array2View<CTYPE> %{Arcane.CNAME##Array2View%}
%typemap(cstype) Arcane::ConstArray2View<CTYPE> %{Arcane.CNAME##ConstArray2View%}
%typemap(ctype, out="Arcane::Array2ViewPOD") Arcane::Array2View<CTYPE > "Arcane::Array2ViewPOD_T<CTYPE >"
%typemap(ctype, out="Arcane::ConstArray2ViewPOD") Arcane::ConstArray2View<CTYPE > "Arcane::ConstArray2ViewPOD_T<CTYPE >"
%typemap(imtype) Arcane::Array2View<CTYPE> %{Arcane.CNAME##Array2View%}
%typemap(imtype) Arcane::ConstArray2View<CTYPE> %{Arcane.CNAME##ConstArray2View%}
%typemap(csin) Arcane::Array2View<CTYPE> "$csinput"
%typemap(csin) Arcane::ConstArray2View<CTYPE> "$csinput"
%typemap(csout) Arcane::Array2View<CTYPE > { return $imcall;  }
%typemap(csout) Arcane::ConstArray2View<CTYPE > { return $imcall;  }
%typemap(in) Arcane::Array2View<CTYPE > %{$1 = Arcane::Array2View<CNAME>($input.m_ptr,$input.m_dim1_size,$input.m_dim2_size); %}
%typemap(in) Arcane::ConstArray2View<CTYPE > %{$1 = Arcane::ConstArray2View<CNAME>($input.m_ptr,$input.m_dim1_size,$input.m_dim2_size); %}
%typemap(out) Arcane::Array2View<CTYPE >
%{
   Arcane::Array2View<CTYPE> result_ref = $1;
   $result .m_ptr = result_ref.unguardedBasePointer();
   $result .m_dim1_size = result_ref.dim1Size();
   $result .m_dim2_size = result_ref.dim2Size();
%}
%typemap(out) Arcane::ConstArray2View<CTYPE >
%{
   Arcane::ConstArray2View<CTYPE> result_ref = $1;
   $result .m_ptr = result_ref.unguardedBasePointer();
   $result .m_dim1_size = result_ref.dim1Size();
   $result .m_dim2_size = result_ref.dim2Size();
%}

namespace Arcane
{
  template<> class ConstArray2View<CTYPE > {
    ConstArray2View() {}
  };
  template<> class Array2View<CTYPE > {
    Array2View() {}
  };
}

%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


SWIG_ARCANE_ARRAYVIEW_SPECIALIZE_NEW2(Arcane::Int16,Int16)
SWIG_ARCANE_ARRAYVIEW_SPECIALIZE_NEW2(Arcane::Int32,Int32)
SWIG_ARCANE_ARRAYVIEW_SPECIALIZE_NEW2(Arcane::Int64,Int64)
SWIG_ARCANE_ARRAYVIEW_SPECIALIZE_NEW2(Arcane::UInt32,UInt32)
SWIG_ARCANE_ARRAYVIEW_SPECIALIZE_NEW2(Arcane::UInt64,UInt64)
SWIG_ARCANE_ARRAYVIEW_SPECIALIZE_NEW2(double,Real)
SWIG_ARCANE_ARRAYVIEW_SPECIALIZE_NEW2(Arcane::Byte,Byte)

SWIG_ARCANE_ARRAYVIEW_SPECIALIZE_NEW2(Arcane::Real2,Real2)
SWIG_ARCANE_ARRAYVIEW_SPECIALIZE_NEW2(Arcane::Real3,Real3)
SWIG_ARCANE_ARRAYVIEW_SPECIALIZE_NEW2(Arcane::Real2x2,Real2x2)
SWIG_ARCANE_ARRAYVIEW_SPECIALIZE_NEW2(Arcane::Real3x3,Real3x3)

SWIG_ARCANE_ARRAY2VIEW_SPECIALIZE_NEW2(Arcane::Byte,Byte)
SWIG_ARCANE_ARRAY2VIEW_SPECIALIZE_NEW2(Arcane::Int16,Int16)
SWIG_ARCANE_ARRAY2VIEW_SPECIALIZE_NEW2(Arcane::Int32,Int32)
SWIG_ARCANE_ARRAY2VIEW_SPECIALIZE_NEW2(Arcane::Int64,Int64)
SWIG_ARCANE_ARRAY2VIEW_SPECIALIZE_NEW2(Arcane::UInt32,UInt32)
SWIG_ARCANE_ARRAY2VIEW_SPECIALIZE_NEW2(Arcane::UInt64,UInt64)
SWIG_ARCANE_ARRAY2VIEW_SPECIALIZE_NEW2(double,Real)

SWIG_ARCANE_ARRAY2VIEW_SPECIALIZE_NEW2(Arcane::Real2,Real2)
SWIG_ARCANE_ARRAY2VIEW_SPECIALIZE_NEW2(Arcane::Real2x2,Real2x2)
SWIG_ARCANE_ARRAY2VIEW_SPECIALIZE_NEW2(Arcane::Real3,Real3)
SWIG_ARCANE_ARRAY2VIEW_SPECIALIZE_NEW2(Arcane::Real3x3,Real3x3)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
