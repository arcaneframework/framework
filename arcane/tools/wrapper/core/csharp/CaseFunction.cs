//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
using System;
using System.Reflection;
using System.Collections.Generic;

#if ARCANE_64BIT
using Integer = System.Int64;
#else
using Integer = System.Int32;
#endif
using Real = System.Double;

namespace Arcane
{
  public class CaseFunctionLoader
  {
    static List<ICaseFunction> m_case_functions;

    /*!
     * \brief Charge les 'CaseFunction' issues de la classe \a class_type.
     *
     * Toutes les méthodes publiques de \a class_type dont le prototype correspond
     * à une \a CaseFunction sont considérées comme des fonctions du jeu de données.
     *
     * Par exemple:
     * \code
     * double func1(int);
     * int func2(double);
     * \endcode
     *
     * Les méthodes trouvées sont ajoutées à \a case_mng
     */
    public static void LoadCaseFunction(ICaseMng case_mng,Type class_type)
    {
      if (class_type==null)
        return;

      ConstructorInfo ci = class_type.GetConstructor(new Type[0]);
      object o = ci.Invoke(null);

      TraceAccessor Trace = new TraceAccessor(case_mng.TraceMng());

      // TODO: utile actuellement pour conserver une référence mais il faudrait ensuite
      // supprimer
      if (m_case_functions==null)
        m_case_functions = new List<ICaseFunction>();

      List<ICaseFunction> current_functions = new List<ICaseFunction>();

      //ICaseMng cm = sd.CaseMng();
      BindingFlags flags = BindingFlags.Public|BindingFlags.InvokeMethod|BindingFlags.Instance|BindingFlags.DeclaredOnly;
      MethodInfo[] methods = class_type.GetMethods(flags);
      
      foreach(MethodInfo method in methods){
        string func_name = method.Name;
        Trace.Info(String.Format("FOUND METHOD name={0}",func_name));
        var cfbi = new CaseFunctionBuildInfo(case_mng.TraceMng(),func_name);
        ICaseFunction ucf = UserCaseFunction.Create(Trace,o,method,cfbi);
        if (ucf==null)
          ucf = UserStandardCaseFunction.Create(Trace,o,method,cfbi);
        if (ucf!=null)
          current_functions.Add(ucf);
        else{
          string msg = $"User function '{func_name}' is invalid."+
          " Valid prototypes are f(Real,Real|Real3) -> Real|Real3 or f(Real|Integer) -> Real|Integer|bool";
          throw new ApplicationException(msg);
        }
      }
      foreach(ICaseFunction cf in current_functions){
        case_mng.AddFunction(cf);
        m_case_functions.Add(cf);
      }
    }
  }

  //! Function f(Real,Real) -> Real
  class RealRealToRealFunctor : IRealRealToRealMathFunctor
  {
    delegate Real MyDelegate(Real a,Real b);
    MyDelegate m_delegate;

    public RealRealToRealFunctor(object o,MethodInfo method)
    {
      m_delegate = (MyDelegate)Delegate.CreateDelegate(typeof(MyDelegate),o,method);
    }
    public override Real Apply(Real a,Real b)
    {
      return m_delegate(a,b);
    }
    public override void Apply(RealConstArrayView a,RealConstArrayView b,RealArrayView out_v)
    {
      for (int i=0; i<a.Length; ++i ){
        out_v[i] = m_delegate(a[i],b[i]);
      }
    }
  }

  //! Function f(Real,Real3) -> Real
  class RealReal3ToRealFunctor : IRealReal3ToRealMathFunctor
  {
    delegate Real MyDelegate(Real a,Real3 b);
    MyDelegate m_delegate;

    public RealReal3ToRealFunctor(object o,MethodInfo method)
    {
      m_delegate = (MyDelegate)Delegate.CreateDelegate(typeof(MyDelegate),o,method);
    }
    public override Real Apply(Real a,Real3 b)
    {
      return m_delegate(a,b);
    }
    public override void Apply(RealConstArrayView a,Real3ConstArrayView b,RealArrayView out_v)
    {
      for (int i=0; i<a.Length; ++i ){
        out_v[i] = m_delegate(a[i],b[i]);
      }
    }
  }

  //! Function f(Real,Real) -> Real3
  class RealRealToReal3Functor : IRealRealToReal3MathFunctor
  {
    delegate Real3 MyDelegate(Real a,Real b);
    MyDelegate m_delegate;

    public RealRealToReal3Functor(object o,MethodInfo method)
    {
      m_delegate = (MyDelegate)Delegate.CreateDelegate(typeof(MyDelegate),o,method);
    }
    public override Real3 Apply(Real a,Real b)
    {
      return m_delegate(a,b);
    }
    public override void Apply(RealConstArrayView a,RealConstArrayView b,Real3ArrayView out_v)
    {
      for (int i=0; i<a.Length; ++i ){
        out_v[i] = m_delegate(a[i],b[i]);
      }
    }
  }

  //! Function f(Real,Real3) -> Real3
  class RealReal3ToReal3Functor : IRealReal3ToReal3MathFunctor
  {
    delegate Real3 MyDelegate(Real a,Real3 b);
    MyDelegate m_delegate;

    public RealReal3ToReal3Functor(object o,MethodInfo method)
    {
      m_delegate = (MyDelegate)Delegate.CreateDelegate(typeof(MyDelegate),o,method);
    }
    public override Real3 Apply(Real a,Real3 b)
    {
      return m_delegate(a,b);
    }
    public override void Apply(RealConstArrayView a,Real3ConstArrayView b,Real3ArrayView out_v)
    {
      for (int i=0; i<a.Length; ++i ){
        out_v[i] = m_delegate(a[i],b[i]);
      }
    }
  }

  //! Représente une fonction étendue du jeu de données (avec un paramètre supplémentaire)
  class UserStandardCaseFunction : Arcane.StandardCaseFunction
  {
    public IRealRealToRealMathFunctor m_rr_to_r;
    public IRealRealToReal3MathFunctor m_rr_to_r3;
    public IRealReal3ToRealMathFunctor m_rr3_to_r;
    public IRealReal3ToReal3MathFunctor m_rr3_to_r3;
    
    UserStandardCaseFunction(IRealRealToRealMathFunctor r,CaseFunctionBuildInfo cfbi) : base(cfbi)
    {
      m_rr_to_r = r;
    }
    UserStandardCaseFunction(IRealRealToReal3MathFunctor r,CaseFunctionBuildInfo cfbi) : base(cfbi)
    {
      m_rr_to_r3 = r;
    }
    UserStandardCaseFunction(IRealReal3ToRealMathFunctor r,CaseFunctionBuildInfo cfbi) : base(cfbi)
    {
      m_rr3_to_r = r;
    }
    UserStandardCaseFunction(IRealReal3ToReal3MathFunctor r,CaseFunctionBuildInfo cfbi) : base(cfbi)
    {
      m_rr3_to_r3 = r;
    }

    public static UserStandardCaseFunction Create(TraceAccessor trace,object o,MethodInfo method,CaseFunctionBuildInfo cfbi)
    {
      string method_name = method.Name;
      ParameterInfo[] method_params = method.GetParameters();
      int nb_param = method_params.Length;

      if (nb_param!=2)
        return null;

      Type real_type = typeof(System.Double);
      Type real3_type = typeof(Arcane.Real3);
      Type return_type = method.ReturnType;
      Type param1_type = method_params[0].ParameterType;
      Type param2_type = method_params[1].ParameterType;

      trace.Info(String.Format("METHOD_INFO={0} f({1},{2}) -> {3}",method_name,param1_type,param2_type,return_type));

      // Créé l'instance correspondante au type de la fonction retournée.
      if (return_type==real_type && param1_type==real_type && param2_type==real_type){
        trace.Info("Found f(Real,Real) -> Real");
        var v = new RealRealToRealFunctor(o,method);
        return new UserStandardCaseFunction(v,cfbi);
      }

      if (return_type==real_type && param1_type==real_type && param2_type==real3_type){
        trace.Info("Found f(Real,Real3) -> Real");
        var v = new RealReal3ToRealFunctor(o,method);
        return new UserStandardCaseFunction(v,cfbi);
      }

      if (return_type==real3_type && param1_type==real_type && param2_type==real_type){
        trace.Info("Found f(Real,Real) -> Real3");
        var v = new RealRealToReal3Functor(o,method);
        return new UserStandardCaseFunction(v,cfbi);
      }

      if (return_type==real3_type && param1_type==real_type && param2_type==real3_type){
        trace.Info("Found f(Real,Real3) -> Real3");
        var v = new RealReal3ToReal3Functor(o,method);
        return new UserStandardCaseFunction(v,cfbi);
      }
      return null;
    }

    public override IRealRealToRealMathFunctor GetFunctorRealRealToReal()
    {
      return m_rr_to_r;
    }

    public override IRealRealToReal3MathFunctor GetFunctorRealRealToReal3()
    {
      return m_rr_to_r3;
    }

    public override IRealReal3ToRealMathFunctor GetFunctorRealReal3ToReal()
    {
      return m_rr3_to_r;
    }

    public override IRealReal3ToReal3MathFunctor GetFunctorRealReal3ToReal3()
    {
      return m_rr3_to_r3;
    }
  }

  /*!
   * \brief Représente une fonction du type table de marche
   *
   * Pour l'instant on a uniquement le support des fonctions
   * avec le prototype suivant: f(Real|Integer) -> Real|Integer|Bool.
   */
  class UserCaseFunction : Arcane.CaseFunction2
  {
    delegate double RealToRealDelegate(double v);
    delegate double IntegerToRealDelegate(int v);
    delegate int RealToIntegerDelegate(double v);
    delegate int IntegerToIntegerDelegate(int v);
    delegate bool RealToBoolDelegate(double v);
    delegate bool IntegerToBoolDelegate(int v);

    RealToRealDelegate m_real_to_real_delegate;
    IntegerToRealDelegate m_integer_to_real_delegate;

    RealToIntegerDelegate m_real_to_integer_delegate;
    IntegerToIntegerDelegate m_integer_to_integer_delegate;

    RealToBoolDelegate m_real_to_bool_delegate;
    IntegerToBoolDelegate m_integer_to_bool_delegate;

    public UserCaseFunction(Arcane.CaseFunctionBuildInfo cfbi) : base(cfbi)
    {
    }

    public override double ValueAsReal(double v)
    {
      if (m_real_to_real_delegate!=null)
        return m_real_to_real_delegate(v);
      if (m_real_to_integer_delegate!=null)
        return (double)m_real_to_integer_delegate(v);
      if (m_integer_to_integer_delegate!=null)
        return (double)m_integer_to_integer_delegate((int)v);
      if (m_integer_to_real_delegate!=null)
        return m_integer_to_integer_delegate((int)v);
      throw new ApplicationException("Bad type for user function. Should be convertible to f(Real) -> Real");
    }

    public override int ValueAsInteger(double v)
    {
      if (m_real_to_integer_delegate!=null)
        return m_real_to_integer_delegate(v);
      if (m_integer_to_integer_delegate!=null)
        return m_integer_to_integer_delegate((int)v);
      if (m_real_to_real_delegate!=null)
        return (int)m_real_to_real_delegate(v);
      if (m_integer_to_real_delegate!=null)
        return (int)m_integer_to_integer_delegate((int)v);
      throw new ApplicationException("Bad type for user function. Should be convertible to f(Real) -> int");
    }

    public override bool ValueAsBool(double v)
    {
      if (m_real_to_bool_delegate!=null)
        return m_real_to_bool_delegate(v);
      if (m_integer_to_bool_delegate!=null)
        return m_integer_to_bool_delegate((int)v);
      throw new ApplicationException("Bad type for user function. Should be convertible to f(Real) -> bool");
    }

    public override double ValueAsReal(int v)
    {
      if (m_integer_to_real_delegate!=null)
        return m_integer_to_integer_delegate(v);
      if (m_real_to_real_delegate!=null)
        return m_real_to_real_delegate((double)v);
      if (m_integer_to_integer_delegate!=null)
        return (double)m_integer_to_integer_delegate(v);
      if (m_real_to_integer_delegate!=null)
        return (double)m_real_to_integer_delegate(v);
      throw new ApplicationException("Bad type for user function. Should be convertible to f(int) -> Real");
    }

    public override int ValueAsInteger(int v)
    {
      if (m_integer_to_integer_delegate!=null)
        return m_integer_to_integer_delegate(v);
      if (m_real_to_integer_delegate!=null)
        return m_real_to_integer_delegate((double)v);
      throw new ApplicationException("Bad type for user function. Should be convertible to f(int) -> int");
    }

    public override bool ValueAsBool(int v)
    {
      if (m_integer_to_bool_delegate!=null)
        return m_integer_to_bool_delegate(v);
      if (m_real_to_bool_delegate!=null)
        return m_real_to_bool_delegate((double)v);
      throw new ApplicationException("Bad type for user function. Should be convertible to f(int) -> bool");
    }

    public override Real3 ValueAsReal3(int v)
    {
      throw new NotSupportedException();
    }

    public override string ValueAsString(int v)
    {
      throw new NotSupportedException();
    }

    public override Real3 ValueAsReal3(Real v)
    {
      throw new NotSupportedException();
    }

    public override string ValueAsString(Real v)
    {
      throw new NotSupportedException();
    }

    public static UserCaseFunction Create(TraceAccessor trace, object o,MethodInfo method,CaseFunctionBuildInfo cfbi)
    {
      string method_name = method.Name;
      ParameterInfo[] method_params = method.GetParameters();
      int nb_param = method_params.Length;

      if (nb_param!=1)
        return null;

      Type real_type = typeof(double);
      Type integer_type = typeof(int);
      Type bool_type = typeof(bool);
      Type return_type = method.ReturnType;
      Type param1_type = method_params[0].ParameterType;

      trace.Info(String.Format("METHOD_INFO={0} f({1}) -> {2}",method_name,param1_type,return_type));

      // Créé l'instance correspondante au type de la fonction retournée

      if (return_type==real_type && param1_type==real_type){
        trace.Info("Found f(Real) -> Real");
        cfbi.m_param_type = ICaseFunction.eParamType.ParamReal;
        cfbi.m_value_type = ICaseFunction.eValueType.ValueReal;
        var v = new UserCaseFunction(cfbi);
        v.m_real_to_real_delegate = (RealToRealDelegate)Delegate.CreateDelegate(typeof(RealToRealDelegate),o,method);
        return v;
      }

      if (return_type==real_type && param1_type==integer_type){
        trace.Info("Found f(int) -> Real");
        cfbi.m_param_type = ICaseFunction.eParamType.ParamInteger;
        cfbi.m_value_type = ICaseFunction.eValueType.ValueReal;
        var v = new UserCaseFunction(cfbi);
        v.m_integer_to_real_delegate = (IntegerToRealDelegate)Delegate.CreateDelegate(typeof(IntegerToRealDelegate),o,method);
        return v;
      }

      if (return_type==integer_type && param1_type==real_type){
        trace.Info("Found f(Real) -> int");
        cfbi.m_param_type = ICaseFunction.eParamType.ParamReal;
        cfbi.m_value_type = ICaseFunction.eValueType.ValueInteger;
        var v = new UserCaseFunction(cfbi);
        v.m_real_to_integer_delegate = (RealToIntegerDelegate)Delegate.CreateDelegate(typeof(RealToIntegerDelegate),o,method);
        return v;
      }

      if (return_type==integer_type && param1_type==integer_type){
        trace.Info("Found f(int) -> int");
        cfbi.m_param_type = ICaseFunction.eParamType.ParamInteger;
        cfbi.m_value_type = ICaseFunction.eValueType.ValueInteger;
        var v = new UserCaseFunction(cfbi);
        v.m_integer_to_integer_delegate = (IntegerToIntegerDelegate)Delegate.CreateDelegate(typeof(IntegerToIntegerDelegate),o,method);
        return v;
      }

      if (return_type==bool_type && param1_type==real_type){
        trace.Info("Found f(Real) -> bool");
        cfbi.m_param_type = ICaseFunction.eParamType.ParamReal;
        cfbi.m_value_type = ICaseFunction.eValueType.ValueBool;
        var v = new UserCaseFunction(cfbi);
        v.m_real_to_bool_delegate = (RealToBoolDelegate)Delegate.CreateDelegate(typeof(RealToBoolDelegate),o,method);
        return v;
      }

      if (return_type==bool_type && param1_type==integer_type){
        trace.Info("Found f(int) -> bool");
        cfbi.m_param_type = ICaseFunction.eParamType.ParamInteger;
        cfbi.m_value_type = ICaseFunction.eValueType.ValueBool;
        var v = new UserCaseFunction(cfbi);
        v.m_integer_to_bool_delegate = (IntegerToBoolDelegate)Delegate.CreateDelegate(typeof(IntegerToBoolDelegate),o,method);
        return v;
      }

      return null;

    }
  }
}

