// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ARCGEOSIM_EXPRESSIONS_FUNCTIONPARSER_H
#define ARCGEOSIM_EXPRESSIONS_FUNCTIONPARSER_H

////////////////////////////////////////////////////////////////////
// TODO : clean the code (suppression du vieux code historique)
//        conversion Arcane du code (nom des types...)       
//        add/transform Doxygen comments
//        
// Idée de perfectionnement
// - Factoriser les buffers d'évaluation par blocking pour l'ensemble des ExpressionBuilder
//   (un pour tous, taillé à la taille max de variable alors)
// - Entrées de type real3
// - Multiple output (multiple real, real3...)
// - Le traitement des fonctions à plusieurs variables doit etre fait via un traitement du nombre variable d'arguments
//   la spécialisation par compte d'arguments pose des problèmes d'annulation de l'arbre syntaxique
///////////////////////////////////////////////////////////////////

// #pragma linked to boost with icc
#ifdef  __INTEL_COMPILER
#pragma warning(disable:383)
#pragma warning(disable:444)
#endif 

#include <boost/spirit/core.hpp>
#include <boost/spirit/attribute.hpp>
#include <boost/spirit/phoenix.hpp>
#include <boost/shared_ptr.hpp>

#include <map>
#include <set>
#include <stack>
#include <typeinfo>
#include <iostream>

// Parametre de blocking
#define BLOCKSIZE 5

#include <arcane/utils/String.h>
#include <arcane/ArcaneException.h>
#include <arcane/utils/TraceInfo.h>

using namespace Arcane;
using namespace boost::spirit;
using namespace phoenix;

#include <cmath>

#include "Numerics/Expressions/IExpressionMng.h"

#include <arcane/utils/ITraceMng.h>

/****************************************************************/

struct FunctionParserData
{
  // Structure pour stocker les elements de type var

  FunctionParserData() 
    : m_global_expression_mng(NULL)
    , m_local_expression_mng(NULL)
    , m_trace_mng(NULL)
  { 
    ;
  }

  enum DataType { PVariable, PConstant, PParameter } ;

  struct VariableRef
  {
    VariableRef() :
      index(-1), order(-1), data(NULL), isOwn(false), isArray(false),
      name(NULL)
    {
    }
    ~VariableRef()
    {
      if (isOwn)
        delete data;
    }
    Integer index; //!< index in register stack
    Integer order; //!< creation order
    Real * data; //!< pointer to data
    bool isOwn; //!< is data own by this reference
    bool isArray; //!< is a array data (or scalar)
    DataType type; //!< kind of data
    Integer size; //!< size of data (if PVariable)
    const std::string * name;
  };
  typedef std::map<std::string, VariableRef> VariableMap;

  mutable VariableMap m_var_map;

  mutable std::vector<const std::string*> m_var_table;
  mutable std::vector<const std::string*> m_cte_table;
  mutable std::vector<const std::string*> m_prm_table;

  // Services utilitaires Arcane
  IExpressionMng * m_global_expression_mng;
  IExpressionMng * m_local_expression_mng;
  ITraceMng * m_trace_mng;

  Real * m_result;
  Integer m_result_size;
};

/****************************************************************/

class DataStack
{
public:
  typedef std::pair<bool, Real *> Operand;

public:
  DataStack() { }
  
  void init(const FunctionParserData::VariableMap & variables, const Integer depth);

public:
  Operand & getOperand(const Integer i)
  {
    return m_stack[i];
  }

  //! set references to variables and constantes
  void prepare(Real * res, const Integer block_index);
private:
  //! Current data stack
  /*! bool param means array when true, single scalar value when false */
  std::vector<Operand> m_stack;

  //! Allocated data stack
  std::vector<Real> m_data;

  const FunctionParserData::VariableMap * m_variables;
};

/****************************************************************/

struct IStackOp
{
  //! virtual destructor
  virtual ~IStackOp()
  {
  }
  //! defaut apply (by block)
  virtual void operator()(DataStack & stack) = 0;
  //! restricted apply (not by block)
  virtual void operator()(DataStack & stack, const Integer n) = 0;
  //! operator depth
  /*! how many value must be un-stacked to apply this operator; -1 means variable or constante */
  virtual Integer size() const = 0;
  //! set operand references
  virtual void setReference(const Integer i, const Integer pos) = 0;
  virtual Integer getReference() const = 0;
};

typedef boost::shared_ptr<IStackOp> IStackOpPtr;

/****************************************************************/

struct define_variable_impl
{
  typedef FunctionParserData::VariableMap Container;
  typedef FunctionParserData::VariableRef Reference;

  define_variable_impl()
  {
  }

  template <typename DataStructure, typename Item> struct result
  {
    typedef void type;
  };

  template<typename DataStructure>
  void operator()(DataStructure & dataStructure, const std::string & item)
  {
    Container & container = dataStructure.m_var_map;
    std::pair<Container::iterator,bool> pos =
      container.insert(Container::value_type(item, Reference()));
    if (pos.second) // new element
      {
        Container::value_type & vref = *pos.first;
        Reference & ref = vref.second;
        ref.order = container.size()-1;
        ref.name = &vref.first;
        ref.type = FunctionParserData::PVariable;
        ref.size = -1;
        dataStructure.m_var_table.push_back(&vref.first);
        dataStructure.m_trace_mng->debug(Trace::Highest) << "New variable : " << item;
      }
    else
      {
        dataStructure.m_trace_mng->debug(Trace::Highest) << "Already existing variable : " << item;
      }
  }
};

function<define_variable_impl> const f_define_variable = define_variable_impl();

struct define_parameter_impl
{
  typedef FunctionParserData::VariableMap Container;
  typedef FunctionParserData::VariableRef Reference;

  define_parameter_impl()
  {
  }

  template <typename DataStructure, typename Item> struct result
  {
    typedef void type;
  };

  template<typename DataStructure>
  void operator()(DataStructure & dataStructure, const std::string & item)
  {
    Container & container = dataStructure.m_var_map;
    std::pair<Container::iterator,bool> pos =
      container.insert(Container::value_type(item, Reference()));
    if (pos.second) // new element
      {
        Container::value_type & vref = *pos.first;
        Reference & ref = vref.second;
        ref.order = container.size()-1;
        ref.name = &vref.first;
        ref.type = FunctionParserData::PParameter;
        dataStructure.m_prm_table.push_back(&vref.first);

        const Real * data = NULL;
        String name = ref.name->c_str();
        if (data == NULL and dataStructure.m_local_expression_mng != NULL)
          data = dataStructure.m_local_expression_mng->constantLookup(name);
        // No lookup in global scope
        // if (data == NULL and container.m_global_expression_mng != NULL)
        //   data = container.m_global_expression_mng->constantLookup(name);
        if (data == NULL)
          {
            ref.size = -1;
            dataStructure.m_trace_mng->debug(Trace::Highest) << "New parameter : " << item;
          }
        else 
          {
            ref.isArray = false;
            ref.data = const_cast<Real*>(data);
            ref.isOwn = false;
            ref.size = 1;
            dataStructure.m_trace_mng->debug(Trace::Highest) << "New parameter : " << item << " with defaut value " << *data;
          }
      }
    else
      {
        dataStructure.m_trace_mng->debug(Trace::Highest) << "Already existing parameter : " << item;
      }
  }
};

function<define_parameter_impl> const f_define_parameter = define_parameter_impl();

struct define_constant_value_impl
{
  typedef FunctionParserData::VariableMap Container;
  typedef FunctionParserData::VariableRef Reference;

  define_constant_value_impl()
  {
  }

  template <typename DataStructure, typename Item> struct result
  {
    typedef void type;
  };

  template<typename DataStructure>
  void operator()(DataStructure & dataStructure, const std::string & value)
  {
    Container & container = dataStructure.m_var_map;
    std::pair<Container::iterator,bool> pos =
      container.insert(Container::value_type(value, Reference()));
    if (pos.second) // new element
      {
        Container::value_type & vref = *pos.first;
        Reference & ref = vref.second;
        Real v = strtod(value.c_str(), NULL);
        ref.isArray = false;
        ref.data = new Real(v);
        ref.isOwn = true;
        ref.order = container.size()-1;
        ref.name = &vref.first;
        ref.type = FunctionParserData::PConstant;
        ref.size = 1;
        dataStructure.m_cte_table.push_back(&vref.first);
        dataStructure.m_trace_mng->debug(Trace::Highest) << "New constant value : " << v;
      }
  }
  
  template<typename DataStructure>
  void operator()(DataStructure & dataStructure, const std::string & value, const Real v)
  {
    Container & container = dataStructure.m_var_map;
    std::pair<Container::iterator,bool> pos =
      container.insert(Container::value_type(value, Reference()));
    if (pos.second) // new element
      {
        Container::value_type & vref = *pos.first;
        Reference & ref = vref.second;
        ref.isArray = false;
        ref.data = new Real(v);
        ref.isOwn = true;
        ref.order = container.size()-1;
        ref.name = &vref.first;
        ref.type = FunctionParserData::PConstant;
        ref.size = 1;
        dataStructure.m_cte_table.push_back(&vref.first);
        dataStructure.m_trace_mng->debug(Trace::Highest) << "New constant value : " << v;
      }
  }
};

function<define_constant_value_impl> const f_define_constant_value = define_constant_value_impl();

/****************************************************************/

struct IRefOp : public IStackOp
{
  virtual ~IRefOp()
  {
  }
  virtual const std::string & name() const = 0;
};

struct VariableRefOp : public IRefOp
{
  VariableRefOp(FunctionParserData::VariableRef & ref) :
    m_ref(ref)
  {
  }
  void operator()(DataStack & stack)
  {
  }
  void operator()(DataStack & stack, const Integer n)
  {
  }
  Integer size() const
  {
    return -1;
  }
  void setReference(const Integer i, const Integer pos)
  {
  }
  Integer getReference() const
  {
    return m_ref.index;
  }
  const std::string & name() const
  {
    return *m_ref.name;
  }
private:
  FunctionParserData::VariableRef & m_ref;
};

struct apply_ref_impl
{
  apply_ref_impl()
  {
  }

  template <typename Container, typename Value> struct result
  {
    typedef void type;
  };

  template <typename Container, typename Value> 
  void operator()(Container & container, Value value)
  {
    typename Container::VariableMap::iterator vpos = container.m_var_map.find(value);
    if (vpos != container.m_var_map.end())
      {
        container.m_op_stack.push_back(IStackOpPtr(new VariableRefOp(vpos->second)));
      }
    else
      { // On cherche la constante nommée 'value' dans 
        // 1- IExpressionMng local
        // 2- IExpressionMng global
        // 3- Erreur sinon
        // Une fois trouvée, sa valeur est ajoutée dans la VariableMap
        
        const Real * data = NULL;
        if (data == NULL and container.m_local_expression_mng != NULL)
          data = container.m_local_expression_mng->constantLookup(value);
        if (data == NULL and container.m_global_expression_mng != NULL)
          data = container.m_global_expression_mng->constantLookup(value);
        
        if (data == NULL)
          container.m_trace_mng->fatal() << "Cannot find constant " << value;

        define_constant_value_impl constant_value;
        constant_value(container,value,*data);
      
        typename Container::VariableMap::iterator cpos = container.m_var_map.find(value);
        if (cpos != container.m_var_map.end())
          {
            container.m_op_stack.push_back(IStackOpPtr(new VariableRefOp(cpos->second)));
          }
        else
          { // vient d'etre ajouté; pourquoi ne pas le trouver
            throw InternalErrorException(A_FUNCINFO,"Inconsistent state while using constant value");
          }
      }
  }
};

function<apply_ref_impl> const f_apply_ref = apply_ref_impl();

/****************************************************************/

template<typename BinOp> struct StackBinOpT : public IStackOp
{
  void operator()(DataStack & stack)
  {
    DataStack::Operand & res_ref = stack.getOperand(m_refs[0]);
    Real * res = res_ref.second;
    // Second argument is first on the stack
    const DataStack::Operand & b_ref = stack.getOperand(m_refs[1]);
    const Real * b = b_ref.second;
    const DataStack::Operand & a_ref = stack.getOperand(m_refs[2]);
    const Real * a = a_ref.second;

    if (a_ref.first) // a:isArray
      if (b_ref.first) // b:isArray
        for (Integer i=0; i<BLOCKSIZE; ++i)
          res[i] = BinOp::apply(a[i], b[i]);
      else
        { // b:isScalar
          const Real b_cte = *b; // save if res alias b
          for (Integer i=0; i<BLOCKSIZE; ++i)
            res[i] = BinOp::apply(a[i], b_cte);
        }
    else
      { // a:isScalar
        if (b_ref.first)
          { // b:isArray
            const Real a_cte = *a; // save if res alias a
            for (Integer i=0; i<BLOCKSIZE; ++i)
              res[i] = BinOp::apply(a_cte, b[i]);
          }
        else
          // b:isScalar
          *res = BinOp::apply(*a, *b);
      }
    res_ref.first = (a_ref.first or b_ref.first); // return type
  }
  void operator()(DataStack & stack, const Integer n)
  {
    DataStack::Operand & res_ref = stack.getOperand(m_refs[0]);
    Real * res = res_ref.second;
    const DataStack::Operand & b_ref = stack.getOperand(m_refs[1]);
    const Real * b = b_ref.second;
    const DataStack::Operand & a_ref = stack.getOperand(m_refs[2]);
    const Real * a = a_ref.second;

    if (a_ref.first) // a:isArray
      if (b_ref.first) // b:isArray
        for (Integer i=0; i<n; ++i)
          res[i] = BinOp::apply(a[i], b[i]);
      else
        { // b:isScalar
          const Real b_cte = *b; // save if res alias b
          for (Integer i=0; i<n; ++i)
            res[i] = BinOp::apply(a[i], b_cte);
        }
    else
      { // a:isScalar
        if (b_ref.first)
          { // b:isArray
            const Real a_cte = *a; // save if res alias a
            for (Integer i=0; i<n; ++i)
              res[i] = BinOp::apply(a_cte, b[i]);
          }
        else
          // b:isScalar
          *res = BinOp::apply(*a, *b);
      }
    res_ref.first = (a_ref.first or b_ref.first); // return type
  }
  Integer size() const
  {
    return 2;
  }
  void setReference(const Integer i, const Integer pos)
  {
    m_refs[i] = pos;
  }
  Integer getReference() const
  {
    return m_refs[0];
  }
private:
  Integer m_refs[3];
};

struct AddOp
{
  inline static Real apply(const Real & a, const Real & b)
  {
    return a+b;
  }
};
struct SubOp
{
  inline static Real apply(const Real & a, const Real & b)
  {
    return a-b;
  }
};
struct MulOp
{
  inline static Real apply(const Real & a, const Real & b)
  {
    return a*b;
  }
};
struct DivOp
{
  inline static Real apply(const Real & a, const Real & b)
  {
    return a/b;
  }
};
struct PowOp
{
  inline static Real apply(const Real & a, const Real & b)
  {
    return std::pow(a, b);
  }
};

/****************************************************************/

template<typename FuncOp> struct StackFuncOpT : public IStackOp
{
  void operator()(DataStack & stack)
  {
    DataStack::Operand & res_ref = stack.getOperand(m_refs[0]);
    Real * res = res_ref.second;
    const DataStack::Operand & a_ref = stack.getOperand(m_refs[1]);
    const Real * a = a_ref.second;
    if (a_ref.first) // a:isArray
      for (Integer i=0; i<BLOCKSIZE; ++i)
        res[i] = FuncOp::apply(a[i]);
    else
      // a:isScalar
      *res = FuncOp::apply(*a);
    res_ref.first = a_ref.first;
  }
  void operator()(DataStack & stack, const Integer n)
  {
    DataStack::Operand & res_ref = stack.getOperand(m_refs[0]);
    Real * res = res_ref.second;
    const DataStack::Operand & a_ref = stack.getOperand(m_refs[1]);
    const Real * a = a_ref.second;
    if (a_ref.first) // a:isArray
      for (Integer i=0; i<n; ++i)
        res[i] = FuncOp::apply(a[i]);
    else
      // a:isScalar
      *res = FuncOp::apply(*a);
    res_ref.first = a_ref.first;
  }
  Integer size() const
  {
    return 1;
  }
  void setReference(const Integer i, const Integer pos)
  {
    m_refs[i] = pos;
  }
  Integer getReference() const
  {
    return m_refs[0];
  }
private:
  Integer m_refs[2];
};

struct CosOp
{
  inline static Real apply(const Real & a)
  {
    return std::cos(a);
  }
};
struct SinOp
{
  inline static Real apply(const Real & a)
  {
    return std::sin(a);
  }
};
struct HeavisideOp
{
  inline static Real apply(const Real & a)
  {
    return (a<0)? 0 : 1;
  }
};
struct SqrtOp
{
  inline static Real apply(const Real & a)
  {
    return std::sqrt(a);
  }
};
struct CopyOp
{
  inline static Real apply(const Real & a)
  {
    return a;
  }
};
struct OppOp
{
  inline static Real apply(const Real & a)
  {
    return -a;
  }
};
struct ExpOp
{
  inline static Real apply(const Real & a)
  {
    return std::exp(a);
  }
};

/****************************************************************/

struct apply_impl
{
  apply_impl()
  {
  }

  template <typename Container, typename ItemOp> struct result
  {
    typedef void type;
  };

  template <typename Container, typename ItemOp> void operator()(Container & container, ItemOp op)
  {
    container.push_back(IStackOpPtr(new ItemOp()));
  }
};

function<apply_impl> const f_apply = apply_impl();

/****************************************************************/

struct apply_func_impl
{
  apply_func_impl()
  {
  }

  template <typename Container, typename ItemOp> struct result
  {
    typedef void type;
  };

  template <typename Container> void operator()(Container & container, const std::string & func)
  {
    if (func == "cos")
      container.m_op_stack.push_back(IStackOpPtr(new StackFuncOpT<CosOp>()));
    else if (func == "sin")
      container.m_op_stack.push_back(IStackOpPtr(new StackFuncOpT<SinOp>()));
    else if (func == "h")
      container.m_op_stack.push_back(IStackOpPtr(new StackFuncOpT<HeavisideOp>()));
    else if (func == "sqrt")
      container.m_op_stack.push_back(IStackOpPtr(new StackFuncOpT<SqrtOp>()));
    else if (func == "exp")
      container.m_op_stack.push_back(IStackOpPtr(new StackFuncOpT<ExpOp>()));
//     else if (func == "pow")
//       container.m_op_stack.push_back(IStackOpPtr(new StackBinOpT<PowOp>()));
    else
      container.m_trace_mng->fatal() << "Cannot find function " << func; // << " with " << n << " args";
  }
};

function<apply_func_impl> const f_apply_func = apply_func_impl();

/****************************************************************/

struct to_print_impl
{
  to_print_impl()
  {
  }

  template <typename DataStructure, typename Item, typename Item2> struct result
  {
    typedef void type;
  };

  template <typename DataStructure, typename Item, typename Item2> 
  void operator()(DataStructure & dataStructure, Item item, Item2 item2)
  {
    dataStructure.m_trace_mng->debug(Trace::Highest) << item << " : " << item2;
  }
};

function<to_print_impl> const f_to_print = to_print_impl();

/****************************************************************/

struct function_closure :
  public boost::spirit::closure<function_closure, std::string>
{
  member1 name;
};

// Structure permettant de stocker les differents parametres permettant de contruire un element
struct global_closure : public boost::spirit::closure<global_closure, Integer>
{
  member1 dummy;
};

/****************************************************************/

// Definition de la grammaire
struct FunctionParser :
  public grammar<FunctionParser, global_closure::context_t>, FunctionParserData
{
  FunctionParser()
  {
    m_depth = 0;
  }

  ~FunctionParser()
  {
    ;
  }

  // Equation textuelle
  mutable std::string m_equation;

  // Taille des variables
  Integer m_depth;

  DataStack m_data_stack;
  mutable std::vector<boost::shared_ptr<IStackOp> > m_op_stack;

  // Definition de la grammaire
  template<typename ScannerT> struct definition
  {
    definition(FunctionParser const & self)
    {
      // Caracteres alpha numeriques
      word = ((alpha_p >> *(alnum_p)));

      lowfactor = ((real_p|int_p)
                   [f_to_print(var(self), "real", construct_<std::string>(arg1,arg2))]
                   [f_define_constant_value( var(self), construct_<std::string>(arg1,arg2) )]
                   [f_apply_ref( var(self), construct_<std::string>(arg1,arg2) )] )
        
        | (function)
        
        | (word
           [f_to_print(var(self), "word", construct_<std::string>(arg1,arg2))]
           [f_apply_ref( var(self), construct_<std::string>(arg1,arg2) )] )
        
        | ((ch_p('-') >> factor)
           [f_to_print(var(self), "operator-","")]
           [f_apply(var(self.m_op_stack),StackFuncOpT<OppOp>())] )
        
        | ('(' >> expression >> ')');
      
      factor = lowfactor >> *( ( (('^' >> factor)
                               [f_to_print(var(self), "operator","^")]
                               [f_apply(var(self.m_op_stack),StackBinOpT<PowOp>())] ) ) );

      term = factor >> *( ( (('*' >> factor)
                             [f_to_print(var(self), "operator","*")]
                             [f_apply(var(self.m_op_stack),StackBinOpT<MulOp>())] )
                            
                            | (('/' >> factor)
                               [f_to_print(var(self), "operator","/")]
                               [f_apply(var(self.m_op_stack),StackBinOpT<DivOp>())] ) ) );

      expression = term >> *( ( (('+' >> term)
                                 [f_to_print(var(self), "operator","+")]
                                 [f_apply(var(self.m_op_stack),StackBinOpT<AddOp>())] )

                                | (('-' >> term)
                                   [f_to_print(var(self), "operator","-")]
                                   [f_apply(var(self.m_op_stack),StackBinOpT<SubOp>())] ) ) );

      function = (word[function.name = construct_<std::string>(arg1,arg2)] >> '(' >> expression >> ')')
        [f_to_print(var(self), "function",function.name)]
        [f_apply_func(var(self),function.name)];

      var_def = word[f_define_variable(var(self), construct_<std::string>(arg1,arg2) )];

      prm_def = word[f_define_parameter(var(self), construct_<std::string>(arg1,arg2) )];

      top = '(' >> (var_def % ch_p(',')) >> !(';' >> (prm_def % ch_p(','))) >> str_p(")->") >> expression;
    }

    ~definition()
    {
      ;
    }

    // Regles de grammaire
    typedef rule<ScannerT, function_closure::context_t> rule_f;
    rule_f function;

    rule<ScannerT> var_def, prm_def;
    rule<ScannerT> expression, top, lowfactor, factor, term, word;

    rule<ScannerT> const& start() const
    {
      return top;
    }
  };

  //! Fonction pour setter la valeur d'une variable
  void setVariable(String st, const Real * x, Integer size);

  //! Fonction pour setter la valeur d'un parametre
  void setParameter(String st, Real x);

  //! Fonction pour setter la valeur d'une constante
  void setConstant(String st, Real x);

  //! Initialisation du vecteur résultat d'évaluation
  void setEvaluationResult(Real * x, Integer size);

  //! Initialisation du vecteur résultat d'évaluation
  void setDerivationResult(String di, Real * x, Integer size);

  //! Initialisation
  void init(IExpressionMng * global_expression_mng, 
            IExpressionMng * local_expression_mng, 
            ITraceMng * trace_mng);

  //! Check integrity and fully qualified evaluator
  bool check() const;

  //! Evaluation
  /*! @return : true si succès
   */
  bool eval();

  //! Nettoyage post-évaluation
  void cleanup();

  //! Fonction pour Parser une chaine de caractere (declaration de cte/var et equation)
  /*! @return : true si succès
   *  \todo envisager un retour sans fatal, avec stockage interne de l'erreur (consultable a posteriori)
   */
  bool parseString(String s);

  Integer getNbVariable() const;

  const std::string& getVariable(const Integer i) const;

  Integer getNbConstant() const;

  const std::string& getConstant(const Integer i) const;

  Integer getNbParameter() const;

  const std::string& getParameter(const Integer i) const;
  
  ITraceMng * getTraceMng() const { return m_trace_mng; }
};

#endif /* ARCGEOSIM_EXPRESSIONS_FUNCTIONPARSER_H */
