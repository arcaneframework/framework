// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#include "Numerics/Expressions/FunctionParser/FunctionParser.h"
#include <sstream>

#include <boost/tokenizer.hpp>

/*---------------------------------------------------------------------------*/

void 
DataStack::
init(const FunctionParserData::VariableMap & variables,
     const Integer depth)
{
  m_variables = &variables;

  m_data.resize(BLOCKSIZE*depth);
  const Integer stack_offset = 1+variables.size();
  m_stack.resize(stack_offset+depth); // result + var + depth
  for (Integer i=0; i<stack_offset; ++i)
    m_stack[i].second = NULL;
  for (Integer i=0; i<depth; ++i)
    m_stack[i+stack_offset].second = &m_data[i*BLOCKSIZE];
}

/*---------------------------------------------------------------------------*/

void 
DataStack::
prepare(Real * res, const Integer block_index)
{
  Integer stack_offset = 0;
  const Integer data_offset = BLOCKSIZE * block_index;
  m_stack[stack_offset].first = true; // pas toujours ...
  m_stack[stack_offset].second = res + data_offset;
  ++stack_offset;

  for (FunctionParserData::VariableMap::const_iterator i= m_variables->begin(); 
       i != m_variables->end(); ++i, ++stack_offset)
    {
      const bool isArray = i->second.isArray;
      m_stack[stack_offset].first = isArray;
      if (isArray)
        m_stack[stack_offset].second = i->second.data + data_offset;
      else
        m_stack[stack_offset].second = i->second.data;
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void 
FunctionParser::
setVariable(String st, const Real * x, Integer size)
{
  std::string s = st.localstr();
  VariableMap::iterator pos = m_var_map.find(s) ;
  if (pos!=m_var_map.end())
    {
      if(pos->second.type != FunctionParserData::PVariable)
        {
          m_trace_mng->fatal() << "Error, trying to assign a non variable";
        }
      pos->second.isArray = true;
      pos->second.data = const_cast<Real *>(x);
      pos->second.size = size;
    }
  else
    {
      m_trace_mng->warning() << "Ttrying to assign a non existing variable";
    }
}

/*---------------------------------------------------------------------------*/

void 
FunctionParser::
setParameter(String st, Real x)
{
  std::string s = st.localstr();
  VariableMap::iterator pos = m_var_map.find(s) ;
  if (pos!=m_var_map.end())
    {
      if(pos->second.type != FunctionParserData::PParameter)
        {
          m_trace_mng->fatal() << "Error, trying to assign a non parameter";
        }
      pos->second.isArray = false;
      pos->second.data = new Real(x);
      pos->second.size = 1;
    }
  else
    {
      m_trace_mng->warning() << "Trying to assign a non existing parameter";
    }
}

/*---------------------------------------------------------------------------*/

void 
FunctionParser::
setConstant(String st, Real x)
{
  std::string s = st.localstr();
  VariableMap::iterator pos = m_var_map.find(s) ;
  if (pos!=m_var_map.end())
    {
      if(pos->second.type != FunctionParserData::PParameter)
        {
          m_trace_mng->fatal() << "Error, trying to assign a non constant";
        }
      pos->second.isArray = false;
      pos->second.data = new Real(x);
      pos->second.size = 1;
    }
  else
    {
      m_trace_mng->warning() << "Trying to assign a non existing constant \n";
    }
}

/*---------------------------------------------------------------------------*/

void 
FunctionParser::
setEvaluationResult(Real * x, Integer size)
{
  m_result = x;
  m_result_size = size;
}

/*---------------------------------------------------------------------------*/

void 
FunctionParser::
setDerivationResult(String di, Real * x, Integer size)
{


}

/*---------------------------------------------------------------------------*/

void 
FunctionParser::
init(IExpressionMng * global_expression_mng, 
     IExpressionMng * local_expression_mng, 
     ITraceMng * trace_mng)
{
  m_global_expression_mng = global_expression_mng;
  m_local_expression_mng = local_expression_mng;
  m_trace_mng = trace_mng;
}

/*---------------------------------------------------------------------------*/

bool
FunctionParser::
check() const
{
  bool ok = true;
  for (VariableMap::iterator i=m_var_map.begin(); i != m_var_map.end(); ++i)
    {
      VariableRef & reference = i->second;
      if (reference.data == NULL) {
        ok = false;
        if (reference.type == PVariable)
          m_trace_mng->warning() << "Not set variable " << i->first;
        else if (reference.type == PParameter)
          m_trace_mng->warning() << "Not set parameter " << i->first;
        else if (reference.type == PConstant)
          m_trace_mng->warning() << "Not set constant " << i->first;
      }
    }
  return ok;
}

/*---------------------------------------------------------------------------*/

bool 
FunctionParser::
eval()
{
  if (not check()) return false;

  // Update variable and constante position
  // this defines order of registers in DataStack
  // check consistancy of variable's size
  Integer data_offset = 1;
  for (VariableMap::iterator i=m_var_map.begin(); i != m_var_map.end(); ++i)
    {
      i->second.index = data_offset++;
      m_trace_mng->debug(Trace::High) << "Virtual register of " << i->first << " " << i->second.index;
    
      if(i->second.type == FunctionParserData::PVariable)
        {
          if(i->second.size != m_result_size)
            {
              m_trace_mng->fatal() << "sizes of variables and result not consistent";
            }
        }
    }

  Integer register_count = 0;
  {
    // for internal data reservation
    std::set<Integer> used_register;
    std::set<Integer> free_register;
    free_register.insert(0); // can use result variable as a register

    // stack variable indexes
    std::stack<Integer> simulated_data_stack;

    // Start simulation
    for (unsigned i=0; i<m_op_stack.size(); ++i)
      {
        IStackOpPtr op = m_op_stack[i];
        const Integer opSize = op->size();

        std::ostringstream oss;
        oss << "Stack " << i << " : " << typeid(*op).name();

        if (opSize < 0) // A reference
          {
            simulated_data_stack.push(op->getReference());
            IRefOp * ref_op = dynamic_cast<IRefOp*>(op.get());
            oss << " ( " << ref_op->name() << " ) -> " << op->getReference();
          }
        else // an operator
          {
            // un-stack arguments
            oss << " ( ";
            for (Integer j=0; j<opSize; ++j)
              {
                Integer index = simulated_data_stack.top();
                oss << index << " ";
                simulated_data_stack.pop();
                op->setReference(j+1, index); // 0 is for result       
                if (used_register.find(index) != used_register.end())
                  {
                    used_register.erase(index);
                    free_register.insert(index);
                  }
              }

            // stack result : may be optimize by using recently freed register
            if (free_register.empty())
              { // add a new register
                free_register.insert(data_offset+register_count);
                register_count++;
              }
            Integer local_result_register = *free_register.begin();
            free_register.erase(local_result_register);
            used_register.insert(local_result_register);
            op->setReference(0, local_result_register);
            simulated_data_stack.push(local_result_register);
            oss << ") -> " << local_result_register;
          }
        m_trace_mng->debug(Trace::High) << oss.str();
      }
  }

  // final result regsiter check
  Integer final_variable_ref = m_op_stack.back()->getReference();
  if (final_variable_ref != 0)
    {
      IStackOpPtr op(new StackFuncOpT<CopyOp>());
      m_trace_mng->debug(Trace::High) << "Stack " << m_op_stack.size() << " : " << typeid(*op).name() << " ( " << final_variable_ref << " ) -> 0";
      m_op_stack.push_back(op);
      op->setReference(1, final_variable_ref);
      op->setReference(0, 0); // result
    }

  m_depth = register_count;
  m_trace_mng->debug(Trace::High) << "Count: vars=" << m_var_map.size() << " ; regs=" << register_count;

//   // clean ReferenceOp from op_data (useless when running stack)
//   std::vector<IStackOpPtr> new_op_stack;
//   for (unsigned i=0; i<m_op_stack.size(); ++i)
//     {
//       IStackOpPtr op = m_op_stack[i];
//       if (op->size() >= 0)
//         new_op_stack.push_back(op);
//     }
//   m_op_stack.swap(new_op_stack);
//   m_trace_mng->debug(Trace::High) << "Cleaned Op Stack size = " << m_op_stack.size();

  m_data_stack.init(m_var_map, m_depth);
  const Integer block_count = m_result_size / BLOCKSIZE;
  const Integer residual_block = m_result_size % BLOCKSIZE;

#warning "TODO PH: Low-level optimization"
//#warning "Inline variable and constant declaration"
//#warning "optimize with  __restrict__"
//#warning "Determine if the result is a scalar at compile time"
//#warning "connect IExpressionMngs for the fonctions accessors; see apply_func"

  { // initial loop for testing scalar/array result
    m_data_stack.prepare(m_result, 0);

    if (block_count>0)
      for (unsigned i=0; i<m_op_stack.size(); ++i)
        // apply operators
        (*m_op_stack[i])(m_data_stack);
    else
      for (unsigned i=0; i<m_op_stack.size(); ++i)
        // apply operators
        (*m_op_stack[i])(m_data_stack, residual_block);

    // si un scalaire est mis dans le resultat final, 
    //   on doit copier cette valeur sur tout le vecteur
    //   dans ce cas, seul le premier block est utile
    DataStack::Operand & result = m_data_stack.getOperand(0);
    if (not result.first) // not a array result
      {
        Real * v = result.second;
        const Real val = m_result[0];
        for(Integer j=1;j<m_result_size;++j)
          v[j] = val;
        return true;
      }
  }

  for (Integer i=1; i<block_count; ++i)
    {
      m_data_stack.prepare(m_result, i);
      for (unsigned i=0; i<m_op_stack.size(); ++i)
        // apply operators
        (*m_op_stack[i])(m_data_stack);
    }

  if (residual_block> 0)
    { // residual block
      m_data_stack.prepare(m_result, block_count);
      for (unsigned i=0; i<m_op_stack.size(); ++i)
        // apply operators
        (*m_op_stack[i])(m_data_stack, residual_block);
    }

  return true;
}

/*---------------------------------------------------------------------------*/

void
FunctionParser::
cleanup()
{


}

/*---------------------------------------------------------------------------*/

bool 
FunctionParser::
parseString(String s)
{
  const parse_info<> info = boost::spirit::parse(s.localstr(),*this >> !end_p, space_p);
  if (info.hit and info.full)
    {
      return true;
    }
  else
    {
      //! \todo envisager un retour sans fatal, avec stockage interne de l'erreur (consultable a posteriori)
      m_trace_mng->fatal() << "Parsing error from >'" << info.stop << "'";
      return false;
    }
}

/*---------------------------------------------------------------------------*/

Integer 
FunctionParser::
getNbVariable() const
{
  return m_var_table.size();
}

/*---------------------------------------------------------------------------*/

const std::string& 
FunctionParser::
getVariable(const Integer i) const
{
  if( i < 0 || i >= (Integer)m_var_table.size() )
    {
      m_trace_mng->fatal() << "No variable with this index";
    }
  return *m_var_table[i];
}

/*---------------------------------------------------------------------------*/

Integer 
FunctionParser::
getNbConstant() const
{
  return m_cte_table.size();
}

/*---------------------------------------------------------------------------*/

const std::string& 
FunctionParser::
getConstant(const Integer i) const
{
  if( i < 0 || i >= (Integer)m_cte_table.size() )
    {
      m_trace_mng->fatal() << "No constant with this index";
    }
  return *m_cte_table[i];
}

/*---------------------------------------------------------------------------*/

Integer 
FunctionParser::
getNbParameter() const
{
  return m_prm_table.size();
}

/*---------------------------------------------------------------------------*/

const std::string& 
FunctionParser::
getParameter(const Integer i) const
{
  if( i < 0 || i >= (Integer)m_prm_table.size() )
    {
      m_trace_mng->fatal() << "No parameter with this index";
    }
  return *m_prm_table[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
