from typing import Set, Tuple, Union, Iterable
from slam.Variables import Variable
from utils.Functions import sort_pair_lists


class BayesTreeNode(object):
    def __init__(self, frontal: Union[Variable, Set[Variable]],
                 separator: Set[Variable] = None,
                 children: Set["BayesTreeNode"] = None,
                 parent: "BayesTreeNode" = None) -> None:
        """
        Create a BayesTreeNode object with no parent clique node
        :param frontal: the set of frontal nodes or
            first frontal node to be added
        :param separator: the set of separator nodes
        :param children: the set of child nodes in the Bayes tree
        :param parent: the parent node
        """
        if isinstance(frontal, Variable):
            self.frontal = {frontal}
        elif type(frontal) is set:
            self.frontal = frontal
        else:
            raise ValueError("The frontal must be either the set of all "
                             "frontal variables, or a frontal variable")
        self.separator = separator if separator else set()
        self.parent = parent if parent else None
        #TODO: do we need an order for children
        self.children = children if children else set()

    def append_child(self, child: "BayesTreeNode") -> "BayesTreeNode":
        """
        Add a child clique node to the current clique node in the Bayes tree
        :param child
        :return: the current clique node
        """
        self.children.add(child)
        child.parent = self
        return self

    def create_child(self, frontal: Variable, separator: Set[Variable] = None
                     ) -> "BayesTreeNode":
        """
        Create a child node
        :param frontal: The first index of frontal nodes
        :param separator: The set of separator nodes
        :return: The created child node
        """
        child = BayesTreeNode(frontal=frontal, separator=separator,
                              children=set())
        self.append_child(child)
        return child

    def set_parent(self, parent: "BayesTreeNode" = None) -> "BayesTreeNode":
        """
        Set the parent clique node
        :param parent
        :return: the current clique node
        """
        self.parent = parent
        if parent is None:
            self.parent = parent
        else:
            parent.append_child(self)
        return self

    def add_frontal(self, frontal: Variable) -> "BayesTreeNode":
        """
        Add a frontal node to the current clique node
        :param frontal
        :return: the current clique node
        """
        self.frontal.add(frontal)
        return self

    def remove_child(self, child: "BayesTreeNode") -> "BayesTreeNode":
        """
        Remove a child clique from children set
        :param child
        :return: the current clique
        """
        self.children.remove(child)
        child.parent = None
        return self

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def vars(self) -> Set[Variable]:
        return self.frontal.union(self.separator)

    @property
    def num_vars(self) -> int:
        return len(self.frontal) + len(self.separator)

    @property
    def dim(self) -> int:
        return sum(var.dim for var in set.union(self.frontal, self.separator))

    @property
    def separator_dim(self) -> int:
        return sum(var.dim for var in self.separator)

    @property
    def frontal_dim(self) -> int:
        return sum(var.dim for var in self.frontal)

    def __str__(self) -> str:
        string = "BayesTreeNode{frontal: " + str(set(var.name for var
                                                     in self.frontal)) + \
                 ", separator: " + str(set(var.name for var in
                                           self.separator)) + ", parent: "
        if self.parent:
            string += "{frontal: " + str(set(var.name for var in
                                             self.parent.frontal)) + \
                ", separator: " + str(set(var.name for var in
                                          self.parent.separator)) + "}, "
        else:
            string += " none, "
        string += "children: "
        if self.children:
            for child in self.children:
                string += "{frontal: " + str(set(var.name for var in
                                                 child.frontal)) + ", separator: " \
                    + str(set(var.name for var in child.separator)) + "}, "
            string = string[:-2]
        else:
            string += " none"
        string += "}"
        return string

    def __copy__(self) -> "BayesTreeNode":
        return BayesTreeNode(frontal=self.frontal.copy(),
                             separator=self.separator.copy(),
                             children=self.children.copy(),
                             parent=self.parent.__copy__()
                             if self.parent else None)

    def copy_without_parents_children(self) -> "BayesTreeNode":
        return BayesTreeNode(frontal=self.frontal.copy(),
                             separator=self.separator.copy())

    def __eq__(self, other: "BayesTreeNode") -> bool:
        """
        Two Bayes tree cliques are deemed equal if both frontal set and
        separator set have the same variables
        """
        return (self.frontal == other.frontal and
                self.separator == other.separator)

    def __hash__(self) -> int:
        return hash((tuple(sorted([var.name for var in self.separator])),
                     tuple(sorted([var.name for var in self.frontal]))))


class BayesTree(object):
    def __init__(self, root_clique: BayesTreeNode = None,
                 frontal: Variable = None) -> None:
        """
        Create a BayesTree object
        :param root_clique: the root clique in the Bayes tree
        :param frontal: the frontal variable
        """
        if root_clique:
            self.root = root_clique
            queue = [self.root]

            # when creating a new tree from a copy of previous root,
            # must re-direct the parent field of its children to the copy.
            for child in root_clique.children:
                child.parent = root_clique

            while queue:
                clique = queue.pop(0)
                if not clique.is_leaf:
                    for child in clique.children:
                        queue.append(child)
        elif frontal:
            self.root = BayesTreeNode(frontal=frontal)
        else:
            raise ValueError("Either the root clique or a root frontal "
                             "variable needs to be specified")
        self.reverse_elimination_order = None

    @property
    def leaves(self) -> Set[BayesTreeNode]:
        leaves = set()
        stack = [self.root]
        while stack:
            clique = stack.pop()
            if clique.children:
                for child in clique.children:
                    stack.append(child)
            else:
                leaves.add(clique)
        return leaves

    @property
    def clique_nodes(self) -> Set[BayesTreeNode]:
        stack = [self.root]
        res = set()
        while stack:
            clique = stack.pop()
            res.add(clique)
            for child in clique.children:
                stack.append(child)
        return res

    def add_node(self, frontal: Variable, parents: Set[Variable] = None
                 ) -> "BayesTree":
        """
        Add a new frontal to the appropriate location in the Bayes tree
        :param frontal:
        :param parents:
        :return: the current Bayes tree
        """
        for clique in self.clique_nodes:
            if parents.issubset(clique.vars):
                if len(parents) == clique.num_vars:
                    clique.add_frontal(frontal)
                else:
                    new_node = clique.create_child(frontal, parents)
                    clique.append_child(new_node)
                break
        return self

    def append_clique(self, clique: BayesTreeNode,
                      parent_clique: BayesTreeNode) -> "BayesTree":
        """
        Attach a clique to the specified clique
        """
        parent_clique.append_child(clique)
        if parent_clique in self.leaves:
            self.leaves.remove(parent_clique)
        stack = [clique]
        while stack:
            c = stack.pop()
            for child in c.children:
                stack.append(child)
        return self

    def __str__(self) -> str:
        string = "BayesTree{"
        queue = [self.root]
        while queue:
            clique = queue.pop(0)
            string += str(clique) + ", "
            for child in clique.children:
                queue.append(child)
        string = string[:-2]
        string += "}"
        return string

    def __copy__(self) -> "BayesTree":
        old_root = self.root
        new_tree = BayesTree(
            root_clique=old_root.copy_without_parents_children())
        if self.reverse_elimination_order:
            new_tree.reverse_elimination_order = \
                self.reverse_elimination_order[:]
        else:
            self.reverse_elimination_order = []
        old_queue = [old_root]
        new_queue = [new_tree.root]
        while old_queue:
            cur_old_clique = old_queue.pop()
            cur_new_clique = new_queue.pop()
            if cur_old_clique.is_leaf:
                self.leaves.add(cur_new_clique)
            else:
                for old_child in cur_old_clique.children:
                    new_child = old_child.copy_without_parents_children()
                    cur_new_clique.append_child(new_child)
                    #looping over all the tree
                    old_queue.append(old_child)
                    new_queue.append(new_child)
        return new_tree

    @property
    def frontal_vars(self) -> Set[Variable]:
        """
        Frontal variables in all cliques
        """
        return set.union(*[c.frontal for c in self.clique_nodes])

    def append_child_bayes_tree(self, child_tree: "BayesTree") -> "BayesTree":
        """
        Attach a child Bayes tree to the current Bayes tree
        :param child_tree
        :return: the current Bayes tree after the addition of child tree
        """
        for attach_point in self.clique_nodes:
            if child_tree.root.separator.issubset(attach_point.vars):
                attach_point.append_child(child_tree.root)
                break
        return self

    def append_child_bayes_trees(self, child_trees: Iterable["BayesTree"]
                                 ) -> "BayesTree":
        for child_tree in child_trees:
            self.append_child_bayes_tree(child_tree)
        return self

    def get_affected_vars_and_partial_bayes_trees(self, vars: Set[Variable]) \
            -> Tuple[Set[Variable], Set["BayesTree"]]:
        """
        Find the Bayes tree cliques that has the specified nodes
        :param vars: the variables that are directly affected
        :return: the affected variables in original Bayes tree, and all other
            sub Bayes trees of unaffected cliques
        """
        # Find all cliques with directly affected variables
        var_to_clique = {}
        for clique in self.clique_nodes:
            for var in clique.frontal:
                var_to_clique[var] = clique
        directly_affected_cliques = set()
        for var in set.intersection(vars, self.frontal_vars):
            directly_affected_cliques.add(var_to_clique[var])

        # Find all affected cliques
        affected_cliques = set()
        for clique in directly_affected_cliques:
            clique_to_visit = clique
            while clique_to_visit and clique_to_visit not in affected_cliques:
                affected_cliques.add(clique_to_visit)
                clique_to_visit = clique_to_visit.parent

        # Construct sub partial trees containing all unaffected cliques
        other_sub_trees = set()
        #TODO: partial_trees seem useless
        partial_trees = BayesTree(self.root.copy_without_parents_children())
        stack_total = [self.root]
        stack_partial = [partial_trees.root]
        while stack_total:
            clique_total = stack_total.pop()
            clique_partial = stack_partial.pop()
            for child in clique_total.children:
                if child in affected_cliques:
                    stack_total.append(child)
                    child_copy = child.copy_without_parents_children()
                    partial_trees.append_clique(child_copy, clique_partial)
                    stack_partial.append(child_copy)
                else:
                    child_copy = child.__copy__()
                    child_copy.parent = None
                    sub_tree = BayesTree(root_clique=child_copy)
                    other_sub_trees.add(sub_tree)
        return (set.union(*[c.frontal for c in affected_cliques]),
                other_sub_trees)

    def clique_variable_pattern(self, clique: BayesTreeNode):
        """
        The pattern is [separator variables, frontal variables] with reverse
        elimination ordering
        """
        separator_list = list(clique.separator)
        frontal_list = list(clique.frontal)
        if separator_list:
            separator_order = [self.reverse_elimination_order.index(var) for var in separator_list]
            separator_order, separator_list = sort_pair_lists(separator_order, separator_list)
        if frontal_list:
            frontal_order = [self.reverse_elimination_order.index(var) for
                             var in frontal_list]
            frontal_order, frontal_list = sort_pair_lists(frontal_order, frontal_list)
        variable_pattern = separator_list + frontal_list
        return variable_pattern

    def clique_ordering(self):
        clique_ordering = []
        queue = [self.root]
        while queue:
            clique = queue.pop(0)
            clique_ordering.append(clique)
            if clique.children:
                for child in clique.children:
                    queue.append(child)
        return clique_ordering
