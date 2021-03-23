// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
namespace Arcane
{
  template<typename InterfaceType>
  class Ref
  {
   private:
    Ref(InterfaceType* t) : m_service(t){}
   public:
    Ref() = default;
    Ref(const Ref<InterfaceType>& rhs) = default;
   public:
    InterfaceType* get();
    static Ref<InterfaceType> create(InterfaceType* t);
    static Ref<InterfaceType> createWithHandle(InterfaceType* t,Internal::ExternalRef h);
  };
}
