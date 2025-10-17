    const express = require('express');
    const app = express();
    const port = 3000; // Or any other available port




    app.get('/', (req, res) => {
        res.send('Hello from your Node.js endpoint!');
    });

    app.get('/api/users', (req, res) => {
        // In a real application, you'd fetch data from a database
        const users = [
            { id: 1, name: 'Alice' },
            { id: 2, name: 'Bob' }
        ];
        res.json(users); // Send JSON response
    });

    // GET with a route parameter
    app.get('/api/users/:id', (req, res) => {
        const userId = req.params.id;
        // Fetch user with userId from database
        res.send(`Fetching user with ID: ${userId}`);
    });

    app.post('/api/users', (req, res) => {
        const newUser = req.body; // Access data sent in the request body
        // Save newUser to a database
        console.log('New user created:', newUser);
        res.status(201).json({ message: 'User created successfully', user: newUser });
    });

    app.listen(port, () => {
        console.log(`Server listening at http://localhost:${port}`);
    });
