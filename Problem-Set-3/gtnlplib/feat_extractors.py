class SimpleFeatureExtractor:

    def get_features(self, parser_state):
        """
        Take in all the parser state information and return features.
        Your features should be autograd.Variable objects of embeddings.

        :param parser_state the ParserState object for the current parse (giving access
            to the stack and input buffer)
        :return A list of autograd.Variable objects, which are the embeddings of your
            features
        """
        # STUDENT
        # hint: parser_state.stack_peek_n
        stack_top = parser_state.stack_peek_n(1)[0]
        buffer_top, buffer_next = parser_state.input_buffer_peek_n(2)
        # print(stack_top, buffer_top, buffer_next)
        return [stack_top.embedding, buffer_top.embedding, buffer_next.embedding]
        # END STUDENT
