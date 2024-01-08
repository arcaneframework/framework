// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ARCANE_SERVICE_OPTIONS_H
#define ARCANE_SERVICE_OPTIONS_H

#include <memory>
#include <iostream>
#include <type_traits>
#include <array>
#include <vector>
#include <tuple>
#include <map>
#include <functional>

namespace StrongOptions
{
template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& o, const std::array<T, N>& a)
{
  for (auto x : a)
    o << " " << x;
  return o << " ";
}

template <typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& a)
{
  for (auto x : a)
    o << " " << x;
  return o << " ";
}

/////// Concat array tools

template <std::size_t... Is> struct seq
{};
template <std::size_t N, std::size_t... Is>
struct gen_seq : gen_seq<N - 1, N - 1, Is...>
{};
template <std::size_t... Is>
struct gen_seq<0, Is...> : seq<Is...>
{};

template <typename T, std::size_t N1, std::size_t... I1, std::size_t N2, std::size_t... I2>
// Expansion pack
std::array<T, N1 + N2>
concat(const std::array<T, N1>& a1, const std::array<T, N2>& a2, seq<I1...>, seq<I2...>)
{
  return std::array<T, N1 + N2>{ { a1[I1] }..., { a2[I2] }... };
}

template <typename T, std::size_t N1, std::size_t N2>
// Initializer for the recursion
std::array<T, N1 + N2>
concat(const std::array<T, N1>& a1, const std::array<T, N2>& a2)
{
  return concat(a1, a2, gen_seq<N1>{}, gen_seq<N2>{});
}

template <typename T, std::size_t N1>
// Initializer for the recursion
std::array<T, N1 + 1>
concat(const std::array<T, N1>& a1, const T& x)
{
  return concat(a1, std::array<T, 1>{ { x } }, gen_seq<N1>{}, gen_seq<1>{});
}

/////////////////

struct OptionTools
{
  ////// requiredFixedArray
  // TODO remove useless copy by std:move from args
  template <typename T, int N, typename... Args>
  static std::array<typename T::type, N>
  requiredFixedArray(Args&&... args)
  {
    return Internal<T, N, 0, Args...>::requiredFixedArray(std::array<typename T::type, 0>(), std::move(args)...);
  }

  template <typename T, int N, int CurN, typename... Args>
  struct Internal
  {};

  template <typename T, int N, int CurN, typename Head, typename... Tail>
  struct Internal<T, N, CurN, Head, Tail...>
  {
    static std::array<typename T::type, N>
    requiredFixedArray(const std::array<typename T::type, CurN>& r, Head&&, Tail&&... args)
    {
      return Internal<T, N, CurN, Tail...>::requiredFixedArray(r, std::move(args)...);
    }
  };

  template <typename T, int N, int CurN, typename... Tail>
  struct Internal<T, N, CurN, T, Tail...>
  {
    static std::array<typename T::type, N>
    requiredFixedArray(const std::array<typename T::type, CurN>& r, T&& t, Tail&&... args)
    {
      return Internal<T, N, CurN + 1, Tail...>::requiredFixedArray(concat(r, t.value), std::move(args)...);
    }
  };

  template <typename T, int N, int CurN>
  struct Internal<T, N, CurN>
  {
    static std::array<typename T::type, N>
    requiredFixedArray(const std::array<typename T::type, CurN>& r)
    {
      static_assert(CurN == N, "Invalid required parameter count");
      return std::move(r);
    }
  };

  /*    ////// requiredVariableArray
    template<typename T, int Nmin, int Nmax, typename... Args> 
    static std::vector<typename T::type>
    requiredVariableArray(Args&&... args) 
    { 
      return Internal2<T,Nmin,Nmax,0,Args...>::requiredVariableArray(std::vector<typename T::type>{}, std::move(args)...);
    }

    template<typename T, int Nmin, int Nmax, int CurN, typename... Args>
    struct Internal2 { };

    template<typename T, int Nmin, int Nmax, int CurN, typename Head, typename... Tail>
    struct Internal2<T,Nmin,Nmax,CurN,Head,Tail...> 
    {
      static std::vector<typename T::type> &&
      requiredVariableArray(std::vector<typename T::type> && r, Head && t, Tail &&... args)
      {
        return Internal2<T,Nmin,Nmax,CurN,Tail...>::requiredVariableArray(std::move(r), std::move(args)...);
      }
    };

    template<typename T, int Nmin, int Nmax, int CurN, typename... Tail>
    struct Internal2<T,Nmin,Nmax,CurN,T,Tail...>
    {
      static std::vector<typename T::type> &&
      requiredVariableArray(std::vector<typename T::type> && r, T && t, Tail &&... args)
      {
        r.emplace_back(t.value);
        return Internal2<T,Nmin,Nmax,CurN+1,Tail...>::requiredVariableArray(std::move(r),std::move(args)...);
      }
    };

    template<typename T, int Nmin, int Nmax, int CurN>
    struct Internal2<T,Nmin,Nmax,CurN>
    {
      static std::vector<typename T::type> &&
      requiredVariableArray(std::vector<typename T::type> && r)
      {
        static_assert(CurN>=Nmin && CurN<=Nmax, "Invalid required parameter count");
        return std::move(r);
      }
    };
*/
  ////// requiredVariableArray
  template <typename T, int Nmin, int Nmax, typename... Args>
  static Arcane::UniqueArray<typename T::type>
  requiredVariableArray(Args&&... args)
  {
    return Internal2<T, Nmin, Nmax, 0, Args...>::requiredVariableArray(Arcane::UniqueArray<typename T::type>{}, std::move(args)...);
  }

  template <typename T, int Nmin, int Nmax, int CurN, typename... Args>
  struct Internal2
  {};

  template <typename T, int Nmin, int Nmax, int CurN, typename Head, typename... Tail>
  struct Internal2<T, Nmin, Nmax, CurN, Head, Tail...>
  {
    static Arcane::UniqueArray<typename T::type>&&
    requiredVariableArray(Arcane::UniqueArray<typename T::type>&& r, [[maybe_unused]] Head&& t, Tail&&... args)
    {
      return Internal2<T, Nmin, Nmax, CurN, Tail...>::requiredVariableArray(std::move(r), std::move(args)...);
    }
  };

  // TODO : Use r.emplace_back when Arcane::Array allows it
  template <typename T, int Nmin, int Nmax, int CurN, typename... Tail>
  struct Internal2<T, Nmin, Nmax, CurN, T, Tail...>
  {
    static Arcane::UniqueArray<typename T::type>&&
    requiredVariableArray(Arcane::UniqueArray<typename T::type>&& r, T&& t, Tail&&... args)
    {
      r.add(t.value);
      return Internal2<T, Nmin, Nmax, CurN + 1, Tail...>::requiredVariableArray(std::move(r), std::move(args)...);
    }
  };

  template <typename T, int Nmin, int Nmax, int CurN>
  struct Internal2<T, Nmin, Nmax, CurN>
  {
    static Arcane::UniqueArray<typename T::type>&&
    requiredVariableArray(Arcane::UniqueArray<typename T::type>&& r)
    {
      static_assert(CurN >= Nmin && (Nmax == -1 || CurN <= Nmax), "Invalid required parameter count");
      return std::move(r);
    }
  };

  ////// optionalSimple
  template <typename T, typename... Args>
  static bool 
  optionalSimple(typename T::type& r, Args&&... args)
  {
    return Internal3<T, 0, Args...>::optionalSimple(r, std::move(args)...);
  }

  template <typename T, int CurN, typename... Args>
  struct Internal3
  {};

  template <typename T, int CurN, typename Head, typename... Tail>
  struct Internal3<T, CurN, Head, Tail...>
  {
    static bool 
    optionalSimple(typename T::type& r, Head&&, Tail&&... args)
    {
      return Internal3<T, CurN, Tail...>::optionalSimple(r, std::move(args)...);
    }
  };

  template <typename T, int CurN, typename... Tail>
  struct Internal3<T, CurN, T, Tail...>
  {
    static bool 
    optionalSimple(typename T::type& r, T&& t, Tail&&... args)
    {
      r = t.value;
      return Internal3<T, CurN + 1, Tail...>::optionalSimple(r, std::move(args)...);
    }
  };

  template <typename T, int CurN>
  struct Internal3<T, CurN>
  {
    static bool 
    optionalSimple([[maybe_unused]] typename T::type& r)
    {
      static_assert(CurN <= 1, "Invalid required parameter count");
      return CurN == 1;
    }
  };

  ////// Restrict
  template <typename AllowedTypes, typename... Args>
  static void 
  checkRestriction(const Args&... args)
  {
    return Internal4<AllowedTypes, Args...>::checkRestriction(args...);
  }

  template <typename AllowedTypes, typename... Args>
  struct Internal4
  {};

  template <typename... AllowedTypes, typename Head, typename... Tail>
  struct Internal4<std::tuple<AllowedTypes...>, Head, Tail...>
  {
    static void 
    checkRestriction([[maybe_unused]] const Head& h, const Tail&... tail)
    {
      static_assert(Internal44<Head, AllowedTypes...>::checkType, "Illegal option");
      return Internal4<std::tuple<AllowedTypes...>, Tail...>::checkRestriction(tail...);
    }
  };

  template <typename... AllowedTypes>
  struct Internal4<std::tuple<AllowedTypes...>>
  {
    static void
    checkRestriction()
    {
      return;
    }
  };

  template <typename Arg, typename... AllowedTypes>
  struct Internal44
  {};

  template <typename Arg, typename Head, typename... Tail>
  struct Internal44<Arg, Head, Tail...>
  {
    static const bool checkType = Internal44<Arg, Head>::checkType || Internal44<Arg, Tail...>::checkType;
  };

  template <typename Arg, typename Head>
  struct Internal44<Arg, Head>
  {
    static const bool checkType = std::is_same<Arg, Head>::value;
  };

  template <typename Arg>
  struct Internal44<Arg>
  {
    static const bool checkType = false;
  };
};

////////////

template <typename Name, typename Type>
struct OptionValue
{
  typedef Name name;
  typedef Type type;
  type value;

  //OptionValue<Name, Type> && operator=(const Type & v) { value = v; return std::move(*this); }
  // friend std::ostream & operator<<(std::ostream & o, const OptionValue<Name, Type> & x) { return o << x.value; }
};

template <typename Name, typename Type>
struct OptionProxy
{
  OptionValue<Name, Type> operator=(const Type&& value) { return OptionValue<Name, Type>{ value }; }
  OptionValue<Name, Type> operator=(const Type& value) { return OptionValue<Name, Type>{ value }; }
  OptionValue<Name, Type> operator=(Type& value) { return OptionValue<Name, Type>{ value }; }

  //OptionValue<Name, Type> operator=(Type && value) {  // possible to optimize in one line as above
  //OptionValue<Name,Type> myopval{std::move(value)};
  //return std::move(myopval);
  //}
};

#define DECLARE_OPTION_EXTERN(name, type) \
  namespace tag \
  { \
    struct name##_t; \
  } \
  typedef OptionValue<tag::name##_t, type> name##_; \
  ARCANE_CORE_EXPORT extern OptionProxy<tag::name##_t, type> _##name;

#define DECLARE_OPTION(name, type) \
  namespace tag \
  { \
    struct name##_t; \
  } \
  typedef OptionValue<tag::name##_t, type> name##_; \
  OptionProxy<tag::name##_t, type> _##name;

}; // namespace StrongOptions

#if !defined(ARCANE_JOIN_HELPER2)
#define ARCANE_JOIN_HELPER2(a, b) a##b
#endif

#if !defined(ARCANE_JOIN_HELPER)
#define ARCANE_JOIN_HELPER(a, b) ARCANE_JOIN_HELPER2(a, b)
#endif

#if !defined(ARCANE_JOIN_WITH_LINE)
#define ARCANE_JOIN_WITH_LINE(a) ARCANE_JOIN_HELPER(a, __LINE__)
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Interface>
class InstanceBuilder
{
 public:

  static std::unique_ptr<Interface>
  create(const std::string& name)
  {
    const std::map<std::string, ctor_functor>& ctor_functors = instance()->m_ctor_functors;
    auto finder = ctor_functors.find(name);
    /*if (finder == ctor_functors.end())
			throw InstanceParameterException{stringer("Cannot find implementation '", name, "' for interface ", typeid(Interface).name())};*/
    return std::unique_ptr<Interface>(finder->second());
  }

 public:

  typedef std::function<Interface*()> ctor_functor;

  static void
  registerImplementation(const std::string& name, const ctor_functor& ctor)
  {
    instance()->m_ctor_functors[name] = ctor;
  }

 private:

  static std::unique_ptr<InstanceBuilder> m_instance;
  static InstanceBuilder<Interface>*
  instance()
  {
    if (!m_instance)
      m_instance.reset(new InstanceBuilder<Interface>());
    return m_instance.get();
  }

 private:

  std::map<std::string, ctor_functor> m_ctor_functors;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Interface, typename Class, typename SrongOptionClass>
struct InstanceRegisterer
{
  InstanceRegisterer(const std::string& name)
  {
    InstanceBuilder<Interface>::registerImplementation(name, []() -> Interface* {
      return new Class(std::move(std::unique_ptr<SrongOptionClass>(new SrongOptionClass{})));
    });
  }
};

#endif //ARCANE_SERVICE_OPTIONS_H
