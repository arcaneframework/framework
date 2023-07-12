using System.Collections.Generic;
using System.Linq;
using System;

namespace GumpCompiler
{
  public partial class Entity
  {
    public Entity ()
    {
      containsField = new List<EmbeddedEntity> ().ToArray ();
      supportsField = new List<Property> ().ToArray ();
    }
  }

  public static class GumpExtension
  {
    public static IEnumerable<Entity> Entities (this gump g)
    {
      return g.model.entity;
    }

    public static IEnumerable<Entity> BaseEntities (this gump g)
    {
      return g.Entities ().Where (e => e.@base == null);
    }

    public static IEnumerable<Entity> DerivedEntities (this gump g)
    {
      return g.Entities ().Where (e => e.@base != null);
    }

    private class PropertyComparer : IEqualityComparer<Property>
    {
      public bool Equals (Property x, Property y)
      {
        return x.name == y.name;
      }

      public int GetHashCode (Property obj)
      {
        return obj.name.GetHashCode ();
      }
    }

    public static IEnumerable<Property> Properties (this gump g)
    {
      var p = g.model.entity.SelectMany (i => i.supports);
      return p.Distinct (new PropertyComparer ());
    }

    public static int NumberOfEntities (this gump g)
    {
      return g.BaseEntities ().Count ();
    }

    public static int NumberOfProperties (this gump g)
    {
      return g.Properties ().Count ();
    }
  }

  public static class PropertyExtensions
  {
    public static string Kind (this Property p)
    {
      return "PK_" + p.name;
    }

    public static string Dim (this Property p)
    {
      switch (p.dim) {
      case PropertyDim.scalar:
        return "Scalar";
      case PropertyDim.vectorial:
        return "Vectorial";
      default:
        throw new Exception ();
      }
    }

    public static string Type (this Property p)
    {
      switch (p.type) {
      case PropertyType.@int:
        return "Integer";
      case PropertyType.real:
        return "Real";
      default:
        throw new Exception ();
      }
    }

    public static IEnumerable<Entity> EntitiesOfProperty (this Property p, gump g)
    {
      // On prend toutes les entités qui ont explicitement la propriété
      var entities = g.Entities ().Where (e => e.supports.Any (c => c.name == p.name));

      // On prend aussi les parents des entités dérivées
      // NB2: avec le recul, ce cas est bizarre mais ne force pas l'utilisation
      var all_entities = new List<Entity> (entities);
      foreach (var e in entities) {
        if (e.@base != null) {
          all_entities.Add (g.Entities ().Single (s => s.name == e.@base));
        }
      }

      return all_entities.Distinct ();
    }
  }

  public static class EntityExtensions
  {
    public static string Object (this Entity e)
    {
      string name = e.name;
      return char.ToLower (name [0]) + name.Substring (1);
    }

    public static string Multiple (this Entity e)
    {
      string name = e.name;
      if (name [name.Length - 1] != 's')
        name += 's';
      return name;
    }

    public static string Enumerator (this Entity e)
    {
      string name = Object (e);
      if (name [name.Length - 1] != 's')
        name += 's';
      return name;
    }

    public static string Tag (this Entity e)
    {
      return "ET_" + e.name;
    }

    public static string Kind (this Entity e)
    {
      return "EK_" + ((e.@base == null) ? e.name : e.@base);
    }

    public static Entity baseEntity (this Entity e, gump g)
    {
      return g.Entities ().FirstOrDefault (i => i.name == e.@base);
    }

    public static IEnumerable<Entity> ChildEntities (this Entity e, gump g)
    {
      var derived = g.Entities ().Where (i => i.@base != null);
      return derived.Where (i => i.@base == e.name);
    }

    public static IEnumerable<Entity> Entities (this Entity e, gump g)
    {
      var container = g.Entities ().Where (i => i.contains != null);
      return container.Where (i => i.contains.Any (c => c.name == e.name));
    }

    public static IEnumerable<Entity> UniqueEntities (this Entity e, gump g)
    {
      var composites = e.contains.Where (i => i.unique == true);
      return g.Entities ().Where (i => composites.Any (c => c.name == i.name));
    }

    public static IEnumerable<IEnumerable<Entity>> UniqueEntitiesByKind (this Entity e, gump g)
    {
      var uniques = e.UniqueEntities (g);
      var unique_kinds = uniques.Select(p => p.Kind ()).Distinct();
      var uniques_list = new List<IEnumerable<Entity>> ();
      foreach(var kind in unique_kinds) {
        uniques_list.Add (uniques.Where(p => p.Kind () == kind));
      }
      return uniques_list;
    }

    public static IEnumerable<Entity> MultipleEntities (this Entity entity, gump g)
    {
      var composites = entity.contains.Where (e => e.unique == false);
      return g.Entities ().Where (e => composites.Any (c => c.name == e.name));
    }

    public static IEnumerable<Entity> AllEntities (this Entity entity, gump g)
    {
      var all = new List<Entity> ();
      all.AddRange (entity.UniqueEntities (g));
      all.AddRange (entity.MultipleEntities (g));
      return all;
    }

    private class EntityComparer : IEqualityComparer<Entity>
    {
      public bool Equals (Entity x, Entity y)
      {
        return x.name == y.name;
      }

      public int GetHashCode (Entity obj)
      {
        return obj.name.GetHashCode ();
      }
    }

    public static IEnumerable<Entity> NestedBaseEntities (this Entity entity, gump g)
    {
      var all = new List<Entity> ();
      foreach (var e in entity.AllEntities(g)) {
        all.Add (e);
        if (e.@base == null) {
          all.AddRange (e.ChildEntities (g));
          all.AddRange (e.NestedBaseEntities (g));
        } else {
          all.Add (e.baseEntity (g));
          all.AddRange (e.baseEntity (g).NestedBaseEntities (g));
        }
      }
      return all.Distinct (new EntityComparer ());
    }
  }
}