// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// typemap pour MutableMemoryView
%typemap(csinterfaces) Arcane::MutableMemoryView "";
%typemap(csbody) Arcane::MutableMemoryView %{ %}
%typemap(SWIG_DISPOSING, methodname="Dispose", methodmodifiers="private") Arcane::MutableMemoryView ""
%typemap(SWIG_DISPOSE, methodname="Dispose", methodmodifiers="private") Arcane::MutableMemoryView ""
%typemap(csclassmodifiers) Arcane::MutableMemoryView "public struct"
%typemap(csattributes) Arcane::MutableMemoryView "[StructLayout(LayoutKind.Sequential)]"
%typemap(cstype) Arcane::MutableMemoryView %{ Arcane.MutableMemoryView %}
%typemap(ctype, out="Arcane::MutableMemoryView",
	 directorout="Arcane::MutableMemoryView",
	 directorin="Arcane::MutableMemoryView") Arcane::MutableMemoryView "Arcane::MutableMemoryView"
%typemap(imtype) Arcane::MutableMemoryView %{ Arcane.MutableMemoryView %}
%typemap(csin) Arcane::MutableMemoryView "$csinput"
%typemap(csout) Arcane::MutableMemoryView { return $imcall;  }
%typemap(in) Arcane::MutableMemoryView %{$1 = Arcane::Arcane::MutableMemoryView($input.m_size,$input.m_ptr); %}
%typemap(directorin) Arcane::MutableMemoryView %{ $input = $1; %}
%typemap(out) Arcane::MutableMemoryView %{ $result = $1; %}
%typemap(directorout) Arcane::MutableMemoryView %{ $1 = $input; %}
%typemap(csdirectorin) Arcane::MutableMemoryView "$iminput"
%typemap(csdirectorout) Arcane::MutableMemoryView "$cscall"
%typemap(cscode) Arcane::MutableMemoryView
%{
    public IntPtr Pointer { get { return m_ptr; } }
    public Int64 ByteSize { get { return m_size; } }
    public Int64 NbElement { get { return m_nb_element; } }

    IntPtr m_ptr;
    Int64 m_size;
    Int64 m_nb_element;
    Int32 m_datatype_size;
%}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
  class MutableMemoryView
  {
    MutableMemoryView() {}
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
