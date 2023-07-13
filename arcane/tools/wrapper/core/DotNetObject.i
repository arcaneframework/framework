// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
%typemap(cscode) Arccore::Internal::ExternalRef
%{
  internal delegate void DestroyDelegate(IntPtr handle);

  [DllImport("$dllimport")]
  static extern void _ArcaneWrapperCoreSetExternalRefDestroyFunctor(DestroyDelegate d);
%}

// La méthode '_SetExternalRefDestroyFunctor' est dans 'arccore_base'.
// Pour éviter d'ajouter une référence explicite à cette bibliothèque, on
// ajoute une fonction dans la bilbiothèque générée par SWIG.
%{
  extern "C" ARCCORE_BASE_EXPORT void
  _SetExternalRefDestroyFunctor(Arccore::Internal::ExternalRef::DestroyFuncType d);
  extern "C" ARCANE_EXPORT void
  _ArcaneWrapperCoreSetExternalRefDestroyFunctor(Arccore::Internal::ExternalRef::DestroyFuncType d)
  {
    _SetExternalRefDestroyFunctor(d);
  }
%}

%typemap(csclassmodifiers) Arccore::Internal::ExternalRef "public partial class"
// La classe 'ExternaRef' est dans 'arccore/base'
namespace Arccore::Internal
{
  class ExternalRef
  {
   public:
    ExternalRef() = default;
    ExternalRef(void* handle) : m_handle(new Handle(handle)){}
   public:
    bool isValid() const { return _internalHandle()!=nullptr; }
    void* _internalHandle() const { return m_handle->handle; }
  };
}
