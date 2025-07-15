// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-

%define ARCANE_SWIG_GENERATE_MEMORY_VIEW(CPP_TYPE,CSHARP_TYPE)
ARCANE_SWIG_GENERATE_POD_TYPE(CPP_TYPE,CSHARP_TYPE)
%typemap(csclassmodifiers) CPP_TYPE "public unsafe struct"
%typemap(directorin) CPP_TYPE %{ $input = $1; %}
%typemap(in) CPP_TYPE %{ $1 = $input; %}
%typemap(out) CPP_TYPE %{ $result = $1; %}
%typemap(directorout) CPP_TYPE %{ $1 = $input; %}
%typemap(csdirectorin) CPP_TYPE "$iminput"
%typemap(csdirectorout) CPP_TYPE "$cscall"
%enddef

 /*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// typemap pour MutableMemoryView
ARCANE_SWIG_GENERATE_MEMORY_VIEW(Arcane::MutableMemoryView,Arcane.MutableMemoryView)
%typemap(cscode) Arcane::MutableMemoryView
%{
    public IntPtr Pointer { get { return m_ptr; } }
    public Int64 ByteSize { get { return m_size; } }
    public Int64 NbElement { get { return m_nb_element; } }
    public Int32 DatatypeSize { get { return m_datatype_size; } }

    IntPtr m_ptr;
    Int64 m_size;
    Int64 m_nb_element;
    Int32 m_datatype_size;
%}

 /*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// typemap pour ConstMemoryView
ARCANE_SWIG_GENERATE_MEMORY_VIEW(Arcane::ConstMemoryView,Arcane.ConstMemoryView)
%typemap(cscode) Arcane::ConstMemoryView
%{
  public IntPtr Pointer { get { return m_ptr; } }
  public Int64 ByteSize { get { return m_size; } }
  public Int64 NbElement { get { return m_nb_element; } }
  public Int32 DatatypeSize { get { return m_datatype_size; } }

  IntPtr m_ptr;
  Int64 m_size;
  Int64 m_nb_element;
  Int32 m_datatype_size;

  internal ConstMemoryView(IntPtr ptr,Int64 size,Int32 nb_element,Int32 datatype_size)
  {
    m_ptr = ptr;
    m_size = size;
    m_nb_element = nb_element;
    m_datatype_size = datatype_size;
  }

  static public ConstMemoryView FromView<T>(ConstArrayView<T> buf)
  where T : unmanaged
  {
    ConstMemoryView v = new ConstMemoryView();
    v.m_ptr = (IntPtr)buf.m_ptr;
    v.m_nb_element = buf.Size;
    v.m_datatype_size = sizeof(T);
    v.m_size = v.m_datatype_size * v.m_nb_element;
    return v;
  }
%}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
  class MutableMemoryView
  {
    MutableMemoryView() {}
  };
  class ConstMemoryView
  {
    ConstMemoryView() {}
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
