import React, { useRef, useEffect } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

const Latex = ({ expression, block = false }) => {
    const containerRef = useRef(null);

    useEffect(() => {
        if (containerRef.current) {
            try {
                katex.render(expression, containerRef.current, {
                    displayMode: block,
                    throwOnError: false,
                    output: 'html'
                });
            } catch (err) {
                console.error("KaTeX error:", err);
                containerRef.current.innerText = expression;
            }
        }
    }, [expression, block]);

    return <span ref={containerRef} />;
};

export default Latex;
