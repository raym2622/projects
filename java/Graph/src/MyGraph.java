import java.util.*;

import com.sun.org.apache.bcel.internal.classfile.Unknown;

/**
 * @Authors (with a partner): Rui Ma, Venkatesh Varada
 *  
 * A representation of a graph.
 * Assumes that we do not have negative cost edges in the graph.
 */
public class MyGraph implements Graph {
	private final Map<Vertex, Set<Edge>> graphs;

    /**
     * Creates a MyGraph object with the given collection of vertices
     * and the given collection of edges.
     * @param v a collection of the vertices in this graph
     * @param e a collection of the edges in this graph
     * @throws IllegalArgumentException if the collection of edges has the
     *  same directed edge more than once with a different weight
     */
    public MyGraph(Collection<Vertex> v, Collection<Edge> e) {
    	graphs = new HashMap<Vertex, Set<Edge>>();
    	for (Vertex ver: v) {
    		graphs.put(ver, new HashSet<Edge>());
    	}
    	// To make sure that same edges with diff weights aren't being added
    	for (Edge edges: e) { 
    		for (Edge edgeAll: e) {
    			if (edgeAll.getSource().equals(edges.getSource()) && 
    					edgeAll.getDestination().equals(edges.getDestination())
    					&& edgeAll.getWeight() != edges.getWeight()) {
    				throw new IllegalArgumentException();
    				
    			}
    		}
    		// Make sure weight is not negative
    		if (v.contains(edges.getSource()) && v.contains(edges.getDestination())
    				&& edges.getWeight() >= 0) { 
    			graphs.get(edges.getSource()).add(edges);
    		}
    	}
    }

    /** 
     * Return the collection of vertices of this graph
     * @return the vertices as a collection (which is anything iterable)
     */
    public Collection<Vertex> vertices() {
    	return graphs.keySet();
    }

    /** 
     * Return the collection of edges of this graph
     * @return the edges as a collection (which is anything iterable)
     */
    public Collection<Edge> edges() {
    	Set<Edge> edges = new HashSet<Edge>();
    	for(Vertex v : graphs.keySet())  {
    		for(Edge e : graphs.get(v)) {
    			edges.add(e);
    		}
    	}
    	return edges;
    }

    /**
     * Return a collection of vertices adjacent to a given vertex v.
     *   i.e., the set of all vertices w where edges v -> w exist in the graph.
     * Return an empty collection if there are no adjacent vertices.
     * @param v one of the vertices in the graph
     * @return an iterable collection of vertices adjacent to v in the graph
     * @throws IllegalArgumentException if v does not exist.
     */
    public Collection<Vertex> adjacentVertices(Vertex v) {
    	if (!graphs.containsKey(v)) {
    		throw new IllegalArgumentException();
    	}
    	Set<Vertex> adjV = new HashSet<Vertex>();
    	for(Edge e : graphs.get(v)) {
    		adjV.add(e.getDestination());
    	}
    	return adjV;
    }

    /**
     * Test whether vertex b is adjacent to vertex a (i.e. a -> b) in a directed graph.
     * Assumes that we do not have negative cost edges in the graph.
     * @param a one vertex
     * @param b another vertex
     * @return cost of edge if there is a directed edge from a to b in the graph, 
     * return -1 otherwise.
     * @throws IllegalArgumentException if a or b do not exist.
     */
    public int edgeCost(Vertex a, Vertex b) {
    	if (!(graphs.containsKey(a) && graphs.containsKey(b))) {
    		throw new IllegalArgumentException();
    	}
    	for(Edge e : graphs.get(a)) {
    		if(e.getDestination().equals(b)) {
    			return e.getWeight();
    		}
    	}
    	return -1;
    }

    /**
     * Returns the shortest path from a to b in the graph, or null if there is
     * no such path.  Assumes all edge weights are nonnegative.
     * Uses Dijkstra's algorithm.
     * @param a the starting vertex
     * @param b the destination vertex
     * @return a Path where the vertices indicate the path from a to b in order
     *   and contains a (first) and b (last) and the cost is the cost of 
     *   the path. Returns null if b is not reachable from a.
     * @throws IllegalArgumentException if a or b does not exist.
     */
    public Path shortestPath(Vertex a, Vertex b) {
 	   if (!(graphs.containsKey(a) && graphs.containsKey(b))) {
 		   throw new IllegalArgumentException();
 	   }
 	   Map<Vertex,Integer> unknown = new HashMap<Vertex,Integer>();
       Set<Vertex> seen = new HashSet<Vertex>(); // known vertices set
       Map<Vertex,Vertex> path = new HashMap<Vertex,Vertex>();
 	   
       unknown.put(a,0); 
       // Prime while loop
 	   while(!unknown.isEmpty()) { 
 		   int min = Integer.MAX_VALUE;
 		   Vertex minVertex = new Vertex("");
 		   
 		   // find the vertex with the min cost and store the cost & vertex
 		   for(Vertex v : unknown.keySet()) { 
 			   if(unknown.get(v) < min) {
 				   min = unknown.get(v);
 				   minVertex = v;
 			   }
 		   }
 		   seen.add(minVertex); 
 		   
 		   // Checks if the shortest path is found
 		   if(minVertex.equals(b)) { 
 			   // the final path
 			   List<Vertex> pathFrom = new ArrayList<Vertex>(); 
 			   
 			   pathFrom.add(b);
 			   while(!pathFrom.get(pathFrom.size()-1).equals(a)) { 
 				   pathFrom.add(path.get(pathFrom.get(pathFrom.size()-1)));
 			   }
 		       // pathFrom is currently in reverse order, so reverse
 			   Collections.reverse(pathFrom); 
 			   return new Path(pathFrom,min);
 		   }
             
 		   for(Vertex v : adjacentVertices(minVertex)) {
 			   if(!seen.contains(v)) {
 				   int cost = min + edgeCost(minVertex,v);   
 				   if(unknown.get(v) == null || cost < unknown.get(v)) { 
 					   unknown.put(v,cost); 
 					   path.put(v,minVertex);
 				   }
 			   }
 		   }
 		   unknown.remove(minVertex);
 	   }
 	   // At this point, no path exists.  
       return null;   
    }
}
