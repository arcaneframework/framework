// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Wrapper pour les classes Real*.
// Definitions pour wrapper les classes:
// - Real2,
// - Real3,
// - Real2x2
// - Real3x3
//
// TODO: le wrapper n'est pas fini sur les opérations possible pour
// ces types.
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
  NOTE:
  Les paramètre de type 'const TYPE&', comme 'const Real3&' en entrée
  de fonction doivent être wrappé en le POD correspondant sans référence
  (ici Real3_POD) sinon ca plante avec mono.
  De plus, pour être correct sous Win32, il faut retourner des types POD,
  donc par exemple Real2_POD au lieu de Real2.

  Donc,
  
  f1(const Real3& v);
  Real3 f2();

  doivent devenir:

  f1(Real3_POD v)
  Real3_POD f2();

  //TODO: il faudra faire les tests sur le comportement en sortie de fonction,
  //par exemple: Real3 f().

  Voir aussi le code de NumericWrapper.h
*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

%include NumericWrapper.h

%define SWIG_ARCANE_NUMERIC_TYPE_NEW(CTYPE,CTYPEPOD)

%typemap(cstype) Arcane::CTYPE %{Arcane.CTYPE%}
%typemap(cstype) const Arcane::CTYPE& %{Arcane.CTYPE%}
%typemap(cstype, out="System.IntPtr", inattributes="ref ") Arcane::CTYPE& %{Arcane.CTYPE%}

%typemap(ctype, out="Arcane::CTYPEPOD") Arcane::CTYPE %{Arcane::CTYPEPOD%}
%typemap(ctype, out="Arcane::CTYPEPOD") const Arcane::CTYPE& %{Arcane::CTYPEPOD%}

%typemap(imtype) Arcane::CTYPE %{Arcane.CTYPE%}
%typemap(imtype) const Arcane::CTYPE& %{Arcane.CTYPE%}
%typemap(imtype, out="System.IntPtr", inattributes="ref ") Arcane::CTYPE& %{Arcane.CTYPE%}

%typemap(csin) Arcane::CTYPE "$csinput"
%typemap(csin) const Arcane::CTYPE& "$csinput"
%typemap(csin) Arcane::CTYPE& "ref $csinput"

%typemap(csout) Arcane::CTYPE { return $imcall;  }
%typemap(csout) const Arcane::CTYPE& { return $imcall;  }
%typemap(csout) Arcane::CTYPE& { return $imcall;  }

%typemap(out) Arcane::CTYPE %{ Arcane::CTYPE* __r = &($1); $result = *(Arcane::CTYPEPOD*)__r; %}
%typemap(out) const Arcane::CTYPE& %{ Arcane::CTYPE* __r = $1; $result = *(Arcane::CTYPEPOD*)__r; %}

%typemap(in) Arcane::CTYPE %{ $1 = $input; %}
%typemap(in) const Arcane::CTYPE& %{ Arcane::CTYPE tmp_$1(_PODToReal($input)); $1 = &tmp_$1; %}

%typemap(directorin) Arcane::CTYPE %{ $input = _RealToPOD($1); %}
%typemap(directorin) const Arcane::CTYPE& %{$input =  (Arcane::CTYPEPOD*) & $1; %}

%typemap(directorout) Arcane::CTYPE %{ $result = _PODToReal($input); %}
%typemap(directorout) const Arcane::CTYPE& %{ $result = _PODToReal($input); %}

%typemap(csdirectorin) Arcane::CTYPE "$iminput"
%typemap(csdirectorin) const Arcane::CTYPE& "$iminput"
%typemap(csdirectorin) Arcane::CTYPE& "ref $iminput"

%typemap(csdirectorout) Arcane::CTYPE "$cscall"
%typemap(csdirectorout) const Arcane::CTYPE& "$cscall"
%typemap(csdirectorout) Arcane::CTYPE& "$cscall"

%enddef

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(SWIGCSHARP)

SWIG_ARCANE_NUMERIC_TYPE_NEW(Real3,Real3_POD)
SWIG_ARCANE_NUMERIC_TYPE_NEW(Real2,Real2_POD)
SWIG_ARCANE_NUMERIC_TYPE_NEW(Real3x3,Real3x3_POD)
SWIG_ARCANE_NUMERIC_TYPE_NEW(Real2x2,Real2x2_POD)

#else

namespace Arcane
{
  class Real3
  {
  public:
    Real x;
    Real y;
    Real z;
  };
  class Real2
  {
  public:
    Real x;
    Real y;
  };
  class Real3x3
  {
  public:
    Real3 x;
    Real3 y;
    Real3 z;
  };
  class Real2x2
  {
  public:
    Real2 x;
    Real2 y;
  };
}

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
