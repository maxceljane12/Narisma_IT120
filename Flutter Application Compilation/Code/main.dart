import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data'; // Add this import for Float32List
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    await Firebase.initializeApp();
    print('Firebase initialized successfully');
  } catch (e) {
    print('Firebase initialization error: $e');
    // Continue app execution even if Firebase fails to initialize
  }
  runApp(const MyApp());
}

// Splash Screen Widget
class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    // Navigate to main app after 3 seconds
    Future.delayed(const Duration(seconds: 3), () {
      if (mounted) {
        Navigator.pushReplacementNamed(context, '/home');
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFF00BCD4), Color(0xFF00ACC1)],
          ),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Bird Logo
            CustomPaint(
              size: const Size(100, 100),
              painter: BirdLogoPainter(color: Colors.white),
            ),
            const SizedBox(height: 30),
            const Text(
              'Bird Classifier',
              style: TextStyle(
                color: Colors.white,
                fontSize: 32,
                fontWeight: FontWeight.bold,
                letterSpacing: 1.2,
              ),
            ),
            const SizedBox(height: 10),
            const Text(
              'AI-Powered Bird Recognition',
              style: TextStyle(
                color: Colors.white70,
                fontSize: 16,
                fontWeight: FontWeight.w400,
              ),
            ),
            const SizedBox(height: 50),
            const CircularProgressIndicator(
              valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
              strokeWidth: 4,
            ),
          ],
        ),
      ),
    );
  }
}

// Custom Bird Logo Painter
class BirdLogoPainter extends CustomPainter {
  final Color color;

  BirdLogoPainter({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    final path = Path();
    
    // Simplified flying bird silhouette
    path.moveTo(size.width * 0.5, size.height * 0.3);
    
    // Left wing
    path.quadraticBezierTo(
      size.width * 0.2, size.height * 0.15,
      size.width * 0.1, size.height * 0.3,
    );
    
    path.quadraticBezierTo(
      size.width * 0.25, size.height * 0.4,
      size.width * 0.4, size.height * 0.45,
    );
    
    // Body
    path.lineTo(size.width * 0.5, size.height * 0.5);
    path.lineTo(size.width * 0.6, size.height * 0.45);
    
    // Right wing
    path.quadraticBezierTo(
      size.width * 0.75, size.height * 0.4,
      size.width * 0.9, size.height * 0.3,
    );
    
    path.quadraticBezierTo(
      size.width * 0.8, size.height * 0.15,
      size.width * 0.5, size.height * 0.3,
    );
    
    path.close();
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(BirdLogoPainter oldDelegate) => color != oldDelegate.color;
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  bool _isDarkMode = false;

  void toggleTheme(bool isDark) {
    setState(() {
      _isDarkMode = isDark;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: const SplashScreen(),
      theme: ThemeData(
        primarySwatch: Colors.teal,
        scaffoldBackgroundColor: const Color(0xFFF0F8FF),
        fontFamily: 'Roboto',
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        primarySwatch: Colors.teal,
        scaffoldBackgroundColor: Colors.grey[900],
        fontFamily: 'Roboto',
        useMaterial3: true,
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF00BCD4),
          foregroundColor: Colors.white,
        ),
        drawerTheme: const DrawerThemeData(
          backgroundColor: Colors.grey,
        ),
        dialogBackgroundColor: Colors.grey[800],
        textTheme: const TextTheme(
          bodyMedium: TextStyle(color: Colors.white),
          bodyLarge: TextStyle(color: Colors.white),
          titleLarge: TextStyle(color: Colors.white),
          titleMedium: TextStyle(color: Colors.white),
        ),
        cardTheme: const CardThemeData(
          color: Color(0xFF424242),
        ),
      ),
      themeMode: _isDarkMode ? ThemeMode.dark : ThemeMode.light,
      onGenerateRoute: (settings) {
        if (settings.name == '/home') {
          return MaterialPageRoute(
            builder: (context) => ImageClassifierPage(isDarkMode: _isDarkMode, toggleTheme: toggleTheme),
          );
        }
        return null;
      },
    );
  }
}

class ImageClassifierPage extends StatefulWidget {
  final bool isDarkMode;
  final Function(bool) toggleTheme;

  const ImageClassifierPage({super.key, required this.isDarkMode, required this.toggleTheme});

  @override
  State<ImageClassifierPage> createState() => _ImageClassifierPageState();
}

class _ImageClassifierPageState extends State<ImageClassifierPage> {
  File? _imageFile;
  String? _prediction;
  int? _predictedIndex;
  double? _confidence;
  List<double>? _allProbabilities;

  final picker = ImagePicker();
  Interpreter? _interpreter;
  List<String>? _labels;
  bool _loading = true;
  bool _modelLoadError = false;
  String _modelErrorMessage = '';

  // Theme mode is now managed in MyApp widget

  // History tracking
  List<Map<String, dynamic>> _classificationHistory = [];
  int _notificationCount = 0;
  bool _showNotifications = true;
  bool _saveResults = true;
  bool _showAnalytics = true;
  
  // Notification system
  List<Map<String, dynamic>> _notifications = [];
  
  // Firebase references
  late FirebaseFirestore _firestore;
  late FirebaseStorage _storage;
  bool _firebaseInitialized = false;

  // Bird descriptions
  final Map<String, String> _birdDescriptions = {
    'Canary': 'Small, brightly colored songbird known for its melodious singing. Canaries are popular pets and come in various colors including yellow, orange, and red. They typically measure 10-15 cm in length and weigh 15-20 grams.',
    'Cockatoo': 'Medium to large parrot with a distinctive crest and curved beak. Known for their intelligence and ability to mimic sounds, cockatoos are social birds native to Australia. They can live 40-60 years and are highly affectionate with their owners.',
    'Finch': 'Small to medium-sized passerine birds with conical bills adapted for seed-eating. Finches are found worldwide and are known for their colorful plumage and cheerful songs. Males often have brighter colors than females.',
    'Flamingo': 'Wading bird with distinctive pink coloration, long legs, and curved bill. Flamingos are filter feeders that get their pink color from their diet of algae and small crustaceans. They can stand on one leg for hours.',
    'Hornbill': 'Tropical bird with a distinctive large, down-curved bill and casque on top. Hornbills are found in Africa and Asia and are known for their unique nesting behavior where females seal themselves in tree holes during breeding.',
    'Kingfisher': 'Small to medium-sized bird with bright plumage and long, sharp beak. Kingfishers dive to catch fish and are found near rivers, lakes, and coastlines worldwide. They can dive at speeds up to 25 mph to catch prey.',
    'Lovebirds': 'Small, colorful parrots known for their strong pair bonds. Lovebirds are social birds native to Africa and are popular pets due to their playful nature. They are 13-17 cm long and known for their acrobatic flight.',
    'Macaw': 'Large, colorful parrot with long tail and strong beak. Macaws are intelligent birds native to Central and South America, known for their ability to mimic human speech. They can live 30-50 years and require large spaces.',
    'Peacock': 'Large bird known for the male\'s iridescent tail with colorful "eye" markings. Peacocks are native to South Asia and are symbols of beauty and royalty. Males fan their tail feathers in elaborate displays to attract females.',
    'Toucan': 'Tropical bird with a large, colorful bill and bright plumage. Toucans are found in Central and South American rainforests and use their bills for feeding and temperature regulation. Despite their large bills, they are lightweight.'
  };

  @override
  void initState() {
    super.initState();
    _initFirebase();
    _loadModel();
  }
  
  // Initialize Firebase
  Future<void> _initFirebase() async {
    try {
      _firestore = FirebaseFirestore.instance;
      _storage = FirebaseStorage.instance;
      _firebaseInitialized = true;
      print('Firebase initialized successfully');
      
      // Add notification if notifications are enabled
      if (_showNotifications) {
        _addNotification(
          title: 'Cloud Services Ready',
          message: 'Connected to Firebase services',
          type: 'success',
        );
      }
    } catch (e) {
      print('Error initializing Firebase: $e');
      
      // Add error notification if notifications are enabled
      if (_showNotifications) {
        _addNotification(
          title: 'Cloud Services Error',
          message: 'Failed to connect to Firebase: ${e.toString()}',
          type: 'error',
        );
      }
    }
  }

  Future<void> _loadModel() async {
    try {
      print("Starting model loading...");
      
      print("Loading model from assets...");
      _interpreter = await Interpreter.fromAsset('assets/models/model_unquant.tflite');
      print("Model loaded successfully");
      
      print("Loading labels...");
      final labelsFile = await rootBundle.loadString('assets/models/labels.txt');
      _labels = labelsFile.split('\n').where((line) => line.trim().isNotEmpty).toList();
      
      // Trim whitespace from each label to ensure clean data
      _labels = _labels!.map((label) => label.trim()).toList();
      
      print("Model and labels loaded successfully");

      setState(() {
        _loading = false;
      });
      
      // Add success notification if notifications are enabled
      if (_showNotifications) {
        _addNotification(
          title: 'Model Loaded',
          message: 'Bird classification model loaded successfully',
          type: 'success',
        );
      }
    } catch (e, stackTrace) {
      print("Error loading model: $e");
      print("Stack trace: $stackTrace");
      // Show error to user
      setState(() {
        _loading = false;
        _modelLoadError = true;
        _modelErrorMessage = 'Failed to load model: ${e.toString()}';
      });
      
      // Add error notification if notifications are enabled
      if (_showNotifications) {
        _addNotification(
          title: 'Model Load Error',
          message: 'Failed to load classification model: ${e.toString()}',
          type: 'error',
        );
      }
      
      // Show a detailed error dialog with retry option
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _showErrorDialog(
          context: context,
          title: 'Model Loading Error',
          message: 'Failed to load the bird classification model. Please check your internet connection and try again.\n\nError: $e',
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(context);
                setState(() {
                  _loading = true;
                  _modelLoadError = false;
                  _modelErrorMessage = '';
                });
                _loadModel();
              },
              child: const Text('Retry'),
            ),
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
          ],
        );
      });
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      print("Picking image from $source");
      final pickedFile = await picker.pickImage(source: source);

      if (pickedFile != null) {
        print("Image picked successfully: ${pickedFile.path}");
        setState(() {
          _imageFile = File(pickedFile.path);
          _prediction = null;
          _predictedIndex = null;
          _confidence = null;
          _allProbabilities = null;
        });

        await _classifyImage(_imageFile!);
              
        // Add to history if save results is enabled
        if (_saveResults && _prediction != null) {
          setState(() {
            _classificationHistory.insert(0, {
              'bird': _prediction,
              'confidence': _confidence,
              'timestamp': DateTime.now(),
              'imagePath': _imageFile!.path,
            });
            if (_classificationHistory.length > 20) {
              _classificationHistory.removeLast();
            }
          });
          
          // Save to Firebase
          _saveResultToFirebase(
            bird: _prediction!,
            confidence: _confidence!,
            imagePath: _imageFile!.path,
          );
          
          // Add success notification if notifications are enabled
          if (_showNotifications) {
            _addNotification(
              title: 'Classification Complete',
              message: 'Successfully identified $_prediction with ${(_confidence! * 100).toStringAsFixed(1)}% confidence',
              type: 'success',
            );
          }
          
          // Show success dialog if enabled
          if (_saveResults) {
            _showSuccessDialog(
              context: context,
              title: 'Classification Successful',
              message: 'The bird has been identified as $_prediction with ${(_confidence! * 100).toStringAsFixed(1)}% confidence.',
            );
          }
        }
      } else {
        print("No image picked");
        // Add info notification if notifications are enabled
        if (_showNotifications) {
          _addNotification(
            title: 'No Image Selected',
            message: 'No image was selected from $source',
            type: 'info',
          );
        }
        
        // Show info dialog
        _showErrorDialog(
          context: context,
          title: 'No Image Selected',
          message: 'You didn\'t select an image from $source. Please try again.',
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('OK'),
            ),
          ],
        );
      }
    } catch (e, stackTrace) {
      print("Error picking image: $e");
      print("Stack trace: $stackTrace");
      // Show error to user
      final errorMessage = 'Error picking image from $source: ${e.toString()}';
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(errorMessage),
          backgroundColor: Colors.red,
          duration: const Duration(seconds: 4),
        ),
      );
      
      // Add error notification if notifications are enabled
      if (_showNotifications) {
        _addNotification(
          title: 'Image Selection Error',
          message: errorMessage,
          type: 'error',
        );
      }
      
      // Show detailed error dialog
      _showErrorDialog(
        context: context,
        title: 'Image Selection Failed',
        message: 'We couldn\'t select an image from $source. This might be due to:\n\n'
            '• Permission denied\n'
            '• Camera not available\n'
            '• Storage access issues\n\n'
            'Please check your device settings and try again.\n\n'
            'Technical details: $e',
      );
    }
  }

  // Add a notification to the notification list
  void _addNotification({
    required String title,
    required String message,
    String type = 'info', // 'success', 'warning', 'error', 'info'
  }) {
    setState(() {
      _notifications.insert(0, {
        'title': title,
        'message': message,
        'timestamp': DateTime.now(),
        'type': type,
      });
      
      // Limit notifications to 50
      if (_notifications.length > 50) {
        _notifications.removeLast();
      }
      
      // Increment notification count
      _notificationCount++;
    });
  }
  
  // Enhanced error dialog with better UX
  void _showErrorDialog({
    required BuildContext context,
    required String title,
    required String message,
    List<Widget>? actions,
  }) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          backgroundColor: widget.isDarkMode ? Colors.grey[800] : Colors.white,
          title: Row(
            children: [
              Icon(Icons.error_outline, color: widget.isDarkMode ? Colors.redAccent : Colors.red, size: 24),
              const SizedBox(width: 12),
              Flexible(
                child: Text(
                  title,
                  style: TextStyle(
                    color: widget.isDarkMode ? Colors.white : Colors.black87,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          content: SizedBox(
            width: MediaQuery.of(context).size.width * 0.8,
            child: Text(
              message,
              style: TextStyle(
                color: widget.isDarkMode ? Colors.grey[300] : Colors.grey[700],
              ),
            ),
          ),
          actions: actions ?? [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text(
                'OK',
                style: TextStyle(
                  color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4),
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        );
      },
    );
  }
  
  // Enhanced success dialog
  void _showSuccessDialog({
    required BuildContext context,
    required String title,
    required String message,
  }) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          backgroundColor: widget.isDarkMode ? Colors.grey[800] : Colors.white,
          title: Row(
            children: [
              Icon(Icons.check_circle, color: widget.isDarkMode ? Colors.green : Colors.green, size: 24),
              const SizedBox(width: 12),
              Flexible(
                child: Text(
                  title,
                  style: TextStyle(
                    color: widget.isDarkMode ? Colors.white : Colors.black87,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          content: SizedBox(
            width: MediaQuery.of(context).size.width * 0.8,
            child: Text(
              message,
              style: TextStyle(
                color: widget.isDarkMode ? Colors.grey[300] : Colors.grey[700],
              ),
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text(
                'OK',
                style: TextStyle(
                  color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4),
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        );
      },
    );
  }
  
  // Save classification result to Firebase Firestore (without Storage uploads)
  Future<void> _saveResultToFirebase({
    required String bird,
    required double confidence,
    required String imagePath,
  }) async {
    if (!_firebaseInitialized) {
      print('Firebase not initialized, skipping save');
      return;
    }
    
    try {
      // Skip image upload to Firebase Storage to avoid billing requirements
      // Just store the image path reference
      String imageUrl = imagePath;
      
      // Save classification data to Firestore
      await _firestore.collection('classifications').add({
        'bird': bird,
        'confidence': confidence,
        'imageUrl': imageUrl,
        'timestamp': FieldValue.serverTimestamp(),
        'deviceInfo': 'Bird Classifier App',
      });
      
      print('Classification saved to Firebase successfully');
      
      // Add success notification if notifications are enabled
      if (_showNotifications) {
        _addNotification(
          title: 'Cloud Sync Complete',
          message: 'Classification saved to cloud database',
          type: 'success',
        );
      }
    } catch (e) {
      print('Error saving to Firebase: $e');
      
      // Add error notification if notifications are enabled
      if (_showNotifications) {
        _addNotification(
          title: 'Cloud Sync Failed',
          message: 'Failed to save classification to cloud: ${e.toString()}',
          type: 'error',
        );
      }
    }
  }

  Future<void> _loadExampleImage(String imageName) async {
    try {
      print("Loading example image: $imageName");
      // Close the drawer first
      Navigator.maybePop(context);
      
      print("Loading asset: assets/examples/$imageName");
      final byteData = await rootBundle.load('assets/examples/$imageName');
      print("Example image loaded from assets, length: ${byteData.lengthInBytes}");
      
      final buffer = byteData.buffer;
      
      // Create a temporary file
      final tempDir = Directory.systemTemp;
      final tempFile = File('${tempDir.path}/$imageName');
      print("Writing to temporary file: ${tempFile.path}");
      await tempFile.writeAsBytes(buffer.asUint8List(byteData.offsetInBytes, byteData.lengthInBytes));
      print("Example image written to temporary file: ${tempFile.path}");
      
      setState(() {
        _imageFile = tempFile;
        _prediction = null;
        _predictedIndex = null;
        _confidence = null;
        _allProbabilities = null;
      });

      await _classifyImage(tempFile);
      
      // Add to history if save results is enabled
      if (_saveResults && _prediction != null) {
        setState(() {
          _classificationHistory.insert(0, {
            'bird': _prediction,
            'confidence': _confidence,
            'timestamp': DateTime.now(),
            'imagePath': tempFile.path,
          });
          if (_classificationHistory.length > 20) {
            _classificationHistory.removeLast();
          }
        });
        
        // Save to Firebase
        _saveResultToFirebase(
          bird: _prediction!,
          confidence: _confidence!,
          imagePath: 'assets/examples/$imageName',
        );
        
        // Add success notification if notifications are enabled
        if (_showNotifications) {
          _addNotification(
            title: 'Classification Complete',
            message: 'Successfully identified $_prediction with ${(_confidence! * 100).toStringAsFixed(1)}% confidence',
            type: 'success',
          );
        }
        
        // Show success dialog if enabled
        if (_saveResults) {
          _showSuccessDialog(
            context: context,
            title: 'Classification Successful',
            message: 'The bird has been identified as $_prediction with ${(_confidence! * 100).toStringAsFixed(1)}% confidence.',
          );
        }
      }
    } catch (e, stackTrace) {
      print("Error loading example image: $e");
      print("Stack trace: $stackTrace");
      // Show error to user
      final errorMessage = 'Error loading example image: ${e.toString()}';
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(errorMessage),
          backgroundColor: Colors.red,
          duration: const Duration(seconds: 4),
        ),
      );
      
      // Reset state to avoid black screen
      setState(() {
        _imageFile = null;
        _prediction = null;
        _predictedIndex = null;
        _confidence = null;
        _allProbabilities = null;
      });
      
      // Add error notification if notifications are enabled
      if (_showNotifications) {
        _addNotification(
          title: 'Example Load Error',
          message: errorMessage,
          type: 'error',
        );
      }
      
      // Show detailed error dialog
      _showErrorDialog(
        context: context,
        title: 'Example Loading Failed',
        message: 'We couldn\'t load the example image. This might be due to:\n\n'
            '• Missing asset files\n'
            '• Corrupted image data\n'
            '• Insufficient storage space\n\n'
            'Please restart the app and try again.\n\n'
            'Technical details: $e',
      );
    }
  }

  Future<void> _classifyImage(File image) async {
    try {
      if (_interpreter == null || _labels == null) {
        print("Interpreter or labels not initialized");
        final errorMessage = 'Model not loaded properly. Please restart the app.';
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(errorMessage),
            backgroundColor: Colors.red,
            duration: const Duration(seconds: 4),
          ),
        );
        
        // Show error dialog
        _showErrorDialog(
          context: context,
          title: 'Model Not Ready',
          message: 'The classification model is not ready yet. Please wait for it to load completely or restart the app if this issue persists.',
        );
        return;
      }

      print("Starting image classification for: ${image.path}");

      // Show progress indicator
      final progressIndicator = AlertDialog(
        backgroundColor: widget.isDarkMode ? Colors.grey[800] : Colors.white,
        content: SizedBox(
          width: MediaQuery.of(context).size.width * 0.8,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const CircularProgressIndicator(
                valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF00BCD4)),
              ),
              const SizedBox(height: 20),
              Text(
                'Analyzing bird image...',
                style: TextStyle(
                  color: widget.isDarkMode ? Colors.white : Colors.black87,
                  fontWeight: FontWeight.w500,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'This may take a few seconds',
                style: TextStyle(
                  color: widget.isDarkMode ? Colors.grey[400] : Colors.grey[600],
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ),
      );
      
      // Show progress dialog
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (BuildContext context) => progressIndicator,
      );

      // Read image bytes
      final imageBytes = image.readAsBytesSync();
      print("Image bytes read: ${imageBytes.length}");

      // Decode image
      final imageData = img.decodeImage(imageBytes);
      if (imageData == null) {
        Navigator.pop(context); // Close progress dialog
        print("Failed to decode image");
        final errorMessage = 'Failed to decode image. Please try another image.';
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(errorMessage),
            backgroundColor: Colors.red,
            duration: const Duration(seconds: 4),
          ),
        );
        
        // Show error dialog
        _showErrorDialog(
          context: context,
          title: 'Image Decode Failed',
          message: 'We couldn\'t process the selected image. Please try:\n\n'
              '• Selecting a different image\n'
              '• Ensuring the image is not corrupted\n'
              '• Checking that the image format is supported (JPG, PNG, etc.)',
        );
        return;
      }
      print("Image decoded successfully. Width: ${imageData.width}, Height: ${imageData.height}");

      // Resize image
      final resizedImage = img.copyResize(imageData, width: 224, height: 224);
      print("Image resized successfully");
      
      // Create image matrix
      final imageMatrix = List.generate(1, (i) => List.generate(224, (j) => List.generate(224, (k) => List<double>.filled(3, 0.0))));
      print("Image matrix initialized");
      
      // Populate image matrix
      for (int x = 0; x < 224; x++) {
        for (int y = 0; y < 224; y++) {
          final pixel = resizedImage.getPixel(x, y);
          // Extract RGB values and normalize to [-1, 1] for Teachable Machine
          imageMatrix[0][x][y][0] = (pixel.r / 127.5) - 1.0;
          imageMatrix[0][x][y][1] = (pixel.g / 127.5) - 1.0;
          imageMatrix[0][x][y][2] = (pixel.b / 127.5) - 1.0;
        }
      }
      print("Image matrix populated");

      // Run inference
      // Convert imageMatrix to Float32List for TensorFlow Lite
      final inputBuffer = Float32List(1 * 224 * 224 * 3);
      int index = 0;
      for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 224; j++) {
          for (int k = 0; k < 224; k++) {
            for (int l = 0; l < 3; l++) {
              inputBuffer[index++] = imageMatrix[i][j][k][l];
            }
          }
        }
      }
      
      // Reshape input buffer to match model input shape [1, 224, 224, 3]
      final input = inputBuffer.buffer.asFloat32List();
      
      // Create output buffer with correct size
      final output = Float32List(10);
      print("Output tensor initialized");
      
      _interpreter!.run(input, output);
      print("Model inference completed");
      
      // Close progress dialog
      Navigator.pop(context);

      // Process results - FIXED TYPE CASTING ISSUE
      // Convert Float32List output to List<double>
      final List<double> rawOutput = List<double>.generate(output.length, (index) => output[index]);
      
      // DEBUG: Print raw model outputs
      print("\n=== RAW MODEL OUTPUT ===");
      for (int i = 0; i < rawOutput.length; i++) {
        print("Class $i (${_labels![i]}): ${rawOutput[i]}");
      }
      print("Raw output sum: ${rawOutput.reduce((a, b) => a + b)}");
      print("========================\n");
      
      // Use raw output as probabilities
      final List<double> probabilities = rawOutput;
      
      final maxProbability = probabilities.reduce((a, b) => a > b ? a : b);
      final predictedIndex = probabilities.indexWhere((element) => element == maxProbability);
      
      print("Predicted: ${_labels![predictedIndex]} with confidence: ${maxProbability}");
      
      // Extract bird name using helper function
      String birdName = _extractBirdName(_labels![predictedIndex]);
      
      // Robust description lookup
      String? description;
      String? matchedKey;
      
      for (var entry in _birdDescriptions.entries) {
        if (entry.key.toLowerCase() == birdName.toLowerCase()) {
          description = entry.value;
          matchedKey = entry.key;
          break;
        }
      }
      
      if (description == null) {
        // Try partial matching if exact match fails
        for (var entry in _birdDescriptions.entries) {
          if (entry.key.toLowerCase().contains(birdName.toLowerCase()) || 
              birdName.toLowerCase().contains(entry.key.toLowerCase())) {
            description = entry.value;
            matchedKey = entry.key;
            break;
          }
        }
      }
      
      if (description == null) {
        description = 'No description available for $birdName';
      }
      
      print("Prediction: Index=$predictedIndex, Label=${_labels![predictedIndex]}, Bird=$birdName, Confidence=${maxProbability * 100}%");

      setState(() {
        _prediction = birdName;
        _predictedIndex = predictedIndex;
        _confidence = maxProbability * 100;
        _allProbabilities = List<double>.from(probabilities);
      });
      
      print("UI updated with prediction results");
    } catch (e, stackTrace) {
      // Close progress dialog if still open
      Navigator.maybePop(context);
      
      print("Error during classification: $e");
      print("Stack trace: $stackTrace");
      // Show error to user
      final errorMessage = 'Classification error: ${e.toString()}';
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(errorMessage),
          backgroundColor: Colors.red,
          duration: const Duration(seconds: 4),
        ),
      );
      
      // Reset state to avoid black screen
      setState(() {
        _prediction = null;
        _predictedIndex = null;
        _confidence = null;
        _allProbabilities = null;
      });
      
      // Show detailed error dialog
      _showErrorDialog(
        context: context,
        title: 'Classification Failed',
        message: 'We couldn\'t classify the bird in your image. This might be due to:\n\n'
            '• Unsupported bird species\n'
            '• Poor image quality\n'
            '• Model processing error\n\n'
            'Please try a different image or restart the app.\n\n'
            'Technical details: $e',
      );
    }
  }

  // Apply softmax function to convert logits to probabilities
  List<double> _applySoftmax(List<double> logits) {
    // Handle empty list
    if (logits.isEmpty) return [];
    
    // Find max value for numerical stability
    final maxLogit = logits.reduce((a, b) => a > b ? a : b);
    
    // Calculate exp for each logit
    final expValues = logits.map((logit) => math.exp(logit - maxLogit)).toList();
    
    // Calculate sum of exp values
    final sumExp = expValues.reduce((a, b) => a + b);
    
    // Avoid division by zero
    if (sumExp == 0) return List<double>.filled(logits.length, 0.0);
    
    // Normalize to get probabilities
    return expValues.map((expVal) => expVal / sumExp).toList();
  }

  String _getImageName(int index) {
    final imageNames = [
      'canary.png',
      'cockatoo.png',
      'finch.png',
      'flamingo.png',
      'hornbill.png',
      'kingfisher.png',
      'lovebirds.png',
      'macaw.png',
      'peacock.png',
      'toucan.png'
    ];
    return imageNames.length > index ? imageNames[index] : 'default.png';
  }

  void _showPickerDialog() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) => Container(
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
        ),
        child: SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                margin: const EdgeInsets.symmetric(vertical: 12),
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey[300],
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(24, 8, 24, 24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Select Image Source',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w600,
                        color: widget.isDarkMode ? Colors.white : Color(0xFF1A1A1A),
                        letterSpacing: -0.5,
                      ),
                    ),
                    const SizedBox(height: 20),
                    _buildPickerOption(
                      icon: Icons.photo_library_outlined,
                      title: 'Photo Library',
                      subtitle: 'Choose from gallery',
                      onTap: () {
                        _pickImage(ImageSource.gallery);
                        Navigator.pop(context);
                      },
                    ),
                    const SizedBox(height: 12),
                    _buildPickerOption(
                      icon: Icons.camera_alt_outlined,
                      title: 'Camera',
                      subtitle: 'Take a new photo',
                      onTap: () {
                        _pickImage(ImageSource.camera);
                        Navigator.pop(context);
                      },
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildPickerOption({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(16),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: widget.isDarkMode ? const Color(0xFF424242) : const Color(0xFFF5F5F5),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF00BCD4), Color(0xFF00ACC1)],
                ),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(icon, color: Colors.white, size: 24),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      color: widget.isDarkMode ? Colors.white : Color(0xFF1A1A1A),
                      letterSpacing: -0.3,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    subtitle,
                    style: TextStyle(
                      fontSize: 13,
                      color: widget.isDarkMode ? Colors.grey[300] : Colors.grey[600],
                    ),
                  ),
                ],
              ),
            ),
            Icon(Icons.chevron_right, color: Colors.grey[400]),
          ],
        ),
      ),
    );
  }

  // Settings functionality removed as per user request

  void _showExamplesBottomSheet() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (_) => Container(
        height: MediaQuery.of(context).size.height * 0.7,
        decoration: BoxDecoration(
          color: widget.isDarkMode ? const Color(0xFF424242) : Colors.white,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
        ),
        child: Column(
          children: [
            Container(
              margin: const EdgeInsets.symmetric(vertical: 12),
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: Colors.grey[300],
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Row(
                children: [
                  const Icon(Icons.photo_library, color: Color(0xFF00BCD4)),
                  const SizedBox(width: 12),
                  const Text(
                    'Example Birds',
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: GridView.count(
                  crossAxisCount: 2,
                  mainAxisSpacing: 16,
                  crossAxisSpacing: 16,
                  children: _buildExampleGridItems(),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  List<Widget> _buildExampleGridItems() {
    final birdNames = [
      'Canary',
      'Cockatoo',
      'Finch',
      'Flamingo',
      'Hornbill',
      'Kingfisher',
      'Lovebirds',
      'Macaw',
      'Peacock',
      'Toucan'
    ];

    return List.generate(10, (index) {
      return Card(
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: Colors.grey[200]!, width: 1),
        ),
        child: InkWell(
          onTap: () => _loadExampleImage(_getImageName(index)),
          borderRadius: BorderRadius.circular(16),
          child: Column(
            children: [
              Expanded(
                child: ClipRRect(
                  borderRadius: const BorderRadius.vertical(top: Radius.circular(16)),
                  child: Image.asset(
                    'assets/examples/${_getImageName(index)}',
                    fit: BoxFit.cover,
                    width: double.infinity,
                    errorBuilder: (context, error, stackTrace) {
                      return Container(
                        color: Colors.grey[100],
                        child: const Icon(Icons.error_outline, size: 40, color: Colors.grey),
                      );
                    },
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(vertical: 10),
                child: Text(
                  birdNames[index],
                  style: TextStyle(
                    color: widget.isDarkMode ? Colors.white : Color(0xFF1A1A1A),
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    letterSpacing: -0.2,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          ),
        ),
      );
    });
  }

  Widget _buildConfidenceIndicator(double confidence) {
    // Determine confidence level and color
    Color indicatorColor;
    String confidenceLevel;
    
    if (confidence >= 80) {
      indicatorColor = const Color(0xFF4CAF50);
      confidenceLevel = 'High';
    } else if (confidence >= 60) {
      indicatorColor = const Color(0xFFFF9800);
      confidenceLevel = 'Medium';
    } else if (confidence >= 40) {
      indicatorColor = const Color(0xFFFFC107);
      confidenceLevel = 'Low';
    } else {
      indicatorColor = const Color(0xFFF44336);
      confidenceLevel = 'Very Low';
    }

    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(
              'Confidence',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w500,
                color: widget.isDarkMode ? Colors.grey[300] : Colors.grey[600],
              ),
            ),
            Text(
              '${confidence.toStringAsFixed(1)}%',
              style: TextStyle(
                fontSize: 28,
                fontWeight: FontWeight.w700,
                color: widget.isDarkMode ? Colors.white : Color(0xFF1A1A1A),
                letterSpacing: -1,
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: Stack(
            children: [
              Container(
                height: 12,
                decoration: BoxDecoration(
                  color: Colors.grey[100],
                ),
              ),
              AnimatedContainer(
                duration: const Duration(milliseconds: 600),
                curve: Curves.easeOutCubic,
                height: 12,
                width: MediaQuery.of(context).size.width * (confidence / 100) * 0.7,
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [indicatorColor, indicatorColor.withOpacity(0.7)],
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),
        Align(
          alignment: Alignment.centerRight,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: indicatorColor.withOpacity(0.1),
              borderRadius: BorderRadius.circular(6),
            ),
            child: Text(
              '$confidenceLevel',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: indicatorColor,
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildTopPredictions() {
    if (_allProbabilities == null || _labels == null) return const SizedBox.shrink();

    // Create a list of (index, probability) pairs and sort by probability
    List<MapEntry<int, double>> indexedProbabilities = [];
    for (int i = 0; i < _allProbabilities!.length; i++) {
      indexedProbabilities.add(MapEntry(i, _allProbabilities![i]));
    }
    
    // Sort by probability in descending order
    indexedProbabilities.sort((a, b) => b.value.compareTo(a.value));
    
    // Take top 3 predictions
    List<MapEntry<int, double>> top3 = indexedProbabilities.take(3).toList();

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: widget.isDarkMode ? const Color(0xFF424242) : Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: widget.isDarkMode ? Colors.black.withOpacity(0.3) : Colors.black.withOpacity(0.05),
            spreadRadius: 0,
            blurRadius: 20,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Color(0xFF00BCD4), Color(0xFF00ACC1)],
                  ),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.leaderboard_rounded, color: Colors.white, size: 20),
              ),
              const SizedBox(width: 12),
              Text(
                'Top Predictions',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w600,
                  color: widget.isDarkMode ? Colors.white : Color(0xFF1A1A1A),
                  letterSpacing: -0.5,
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          ...top3.asMap().entries.map((entry) {
            int index = entry.value.key;
            double probability = entry.value.value * 100;
            // Extract bird name using helper function
            String birdName = _extractBirdName(_labels![index]);
            
            // Debug print to see what names are being extracted
            print("DEBUG - Index: $index");
            print("DEBUG - Full label: ${_labels![index]}");
            print("DEBUG - Extracted bird name: '$birdName'");
            print("DEBUG - Bird descriptions keys: ${_birdDescriptions.keys.toList()}");
            
            // Robust description lookup
            String? description;
            String? matchedKey;
            
            for (var entry in _birdDescriptions.entries) {
              if (entry.key.toLowerCase() == birdName.toLowerCase()) {
                description = entry.value;
                matchedKey = entry.key;
                break;
              }
            }
            
            if (description == null) {
              // Try partial matching if exact match fails
              for (var entry in _birdDescriptions.entries) {
                if (entry.key.toLowerCase().contains(birdName.toLowerCase()) || 
                    birdName.toLowerCase().contains(entry.key.toLowerCase())) {
                  description = entry.value;
                  matchedKey = entry.key;
                  print("Found partial match in top predictions: '$matchedKey' for '$birdName'");
                  break;
                }
              }
            }
            
            if (description == null) {
              description = 'No description available for $birdName';
              print("No match found in top predictions for '$birdName'");
            } else {
              print("Matched description in top predictions for '$birdName' using key '$matchedKey'");
            }
            
            print("DEBUG - Final description: '$description'");
            
            return Padding(
              padding: const EdgeInsets.only(bottom: 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        width: 32,
                        height: 32,
                        decoration: BoxDecoration(
                          gradient: index == _predictedIndex 
                              ? const LinearGradient(
                                  colors: [Color(0xFF00BCD4), Color(0xFF00ACC1)],
                                )
                              : null,
                          color: index == _predictedIndex ? null : Colors.grey[200],
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Center(
                          child: Text(
                            '${entry.key + 1}',
                            style: TextStyle(
                              color: index == _predictedIndex ? Colors.white : Colors.grey[600],
                              fontWeight: FontWeight.w600,
                              fontSize: 14,
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          birdName,
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: index == _predictedIndex ? FontWeight.w600 : FontWeight.w400,
                            color: index == _predictedIndex ? (widget.isDarkMode ? Colors.white : const Color(0xFF1A1A1A)) : (widget.isDarkMode ? Colors.grey[300] : const Color(0xFF666666)),
                            letterSpacing: -0.3,
                          ),
                        ),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: index == _predictedIndex 
                              ? const Color(0xFF00BCD4).withOpacity(0.1)
                              : Colors.grey[100],
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Text(
                          '${probability.toStringAsFixed(1)}%',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: index == _predictedIndex ? const Color(0xFF00BCD4) : Colors.grey[700],
                          ),
                        ),
                      ),
                    ],
                  ),
                  // Show detailed description only for the top 1 prediction
                  if (entry.key == 0)
                    Padding(
                      padding: const EdgeInsets.only(top: 12),
                      child: Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: widget.isDarkMode ? const Color(0xFF555555) : const Color(0xFFF5F5F5),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Characteristics:',
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w600,
                                color: widget.isDarkMode ? Colors.white : Color(0xFF1A1A1A),
                                letterSpacing: -0.2,
                              ),
                            ),
                            const SizedBox(height: 6),
                            Text(
                              description,
                              style: TextStyle(
                                fontSize: 13,
                                color: widget.isDarkMode ? Colors.grey[300] : Colors.grey[700],
                                height: 1.5,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                ],
              ),
            );
          }).toList(),
        ],
      ),
    );
  }

  // Helper function to extract bird name from label
  String _extractBirdName(String label) {
    // Split by space and take everything after the first part (which is the index)
    List<String> parts = label.split(' ');
    if (parts.length > 1) {
      // Join all parts except the first one (index)
      String birdName = parts.sublist(1).join(' ').trim();
      print("Extracted bird name from '$label': '$birdName'");
      return birdName;
    } else {
      // If there's only one part, return it as is
      String birdName = parts[0].trim();
      print("Single part bird name from '$label': '$birdName'");
      return birdName;
    }
  }

  // Custom bird logo widget
  Widget _buildBirdLogo({double size = 40, Color color = Colors.white}) {
    return CustomPaint(
      size: Size(size, size),
      painter: BirdLogoPainter(color: color),
    );
  }

  void _showSettingsDialog() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return StatefulBuilder(
          builder: (BuildContext context, StateSetter setState) {
            return AlertDialog(
              backgroundColor: widget.isDarkMode ? Colors.grey[800] : Colors.white,
              title: Row(
                children: [
                  Icon(Icons.settings, color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4)),
                  const SizedBox(width: 12),
                  Text(
                    'Settings',
                    style: TextStyle(
                      color: widget.isDarkMode ? Colors.white : const Color(0xFF00BCD4),
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.palette, color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4)),
                      const SizedBox(width: 12),
                      Text(
                        'Appearance',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                          color: widget.isDarkMode ? Colors.white : Colors.black87,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Row(
                        children: [
                          Icon(Icons.dark_mode, color: widget.isDarkMode ? Colors.white70 : Colors.black54),
                          const SizedBox(width: 8),
                          Text(
                            'Dark Mode',
                            style: TextStyle(
                              fontSize: 16,
                              color: widget.isDarkMode ? Colors.white : Colors.black87,
                            ),
                          ),
                        ],
                      ),
                      Switch(
                        value: widget.isDarkMode,
                        onChanged: (bool value) {
                          setState(() {
                            widget.toggleTheme(value);
                          });
                        },
                        activeColor: const Color(0xFF00BCD4),
                        inactiveThumbColor: Colors.grey[300],
                        inactiveTrackColor: Colors.grey[400],
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    widget.isDarkMode 
                        ? 'Dark mode with gray background and white text' 
                        : 'Light mode with default background',
                    style: TextStyle(
                      fontSize: 12,
                      color: widget.isDarkMode ? Colors.grey[400] : Colors.grey[600],
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Row(
                        children: [
                          Icon(Icons.analytics, color: widget.isDarkMode ? Colors.white70 : Colors.black54),
                          const SizedBox(width: 8),
                          Text(
                            'Show Analytics Graph',
                            style: TextStyle(
                              fontSize: 16,
                              color: widget.isDarkMode ? Colors.white : Colors.black87,
                            ),
                          ),
                        ],
                      ),
                      Switch(
                        value: _showAnalytics,
                        onChanged: (bool value) {
                          setState(() {
                            _showAnalytics = value;
                          });
                        },
                        activeColor: const Color(0xFF00BCD4),
                        inactiveThumbColor: Colors.grey[300],
                        inactiveTrackColor: Colors.grey[400],
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _showAnalytics 
                        ? 'Display analytics graph after classification' 
                        : 'Hide analytics graph after classification',
                    style: TextStyle(
                      fontSize: 12,
                      color: widget.isDarkMode ? Colors.grey[400] : Colors.grey[600],
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Row(
                        children: [
                          Icon(Icons.notifications, color: widget.isDarkMode ? Colors.white70 : Colors.black54),
                          const SizedBox(width: 8),
                          Text(
                            'Enable Notifications',
                            style: TextStyle(
                              fontSize: 16,
                              color: widget.isDarkMode ? Colors.white : Colors.black87,
                            ),
                          ),
                        ],
                      ),
                      Switch(
                        value: _showNotifications,
                        onChanged: (bool value) {
                          setState(() {
                            _showNotifications = value;
                          });
                        },
                        activeColor: const Color(0xFF00BCD4),
                        inactiveThumbColor: Colors.grey[300],
                        inactiveTrackColor: Colors.grey[400],
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _showNotifications 
                        ? 'Receive notifications for new features and updates' 
                        : 'Notifications are disabled',
                    style: TextStyle(
                      fontSize: 12,
                      color: widget.isDarkMode ? Colors.grey[400] : Colors.grey[600],
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Row(
                        children: [
                          Icon(Icons.save, color: widget.isDarkMode ? Colors.white70 : Colors.black54),
                          const SizedBox(width: 8),
                          Text(
                            'Save Results to History',
                            style: TextStyle(
                              fontSize: 16,
                              color: widget.isDarkMode ? Colors.white : Colors.black87,
                            ),
                          ),
                        ],
                      ),
                      Switch(
                        value: _saveResults,
                        onChanged: (bool value) {
                          setState(() {
                            _saveResults = value;
                          });
                        },
                        activeColor: const Color(0xFF00BCD4),
                        inactiveThumbColor: Colors.grey[300],
                        inactiveTrackColor: Colors.grey[400],
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _saveResults 
                        ? 'Save classification results to history' 
                        : 'Do not save results to history',
                    style: TextStyle(
                      fontSize: 12,
                      color: widget.isDarkMode ? Colors.grey[400] : Colors.grey[600],
                    ),
                  ),
                ],
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: Text(
                    'Close',
                    style: TextStyle(
                      color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4), 
                      fontSize: 16, 
                      fontWeight: FontWeight.bold
                    ),
                  ),
                ),
              ],
            );
          },
        );
      },
    );
  }

  void _showNotificationsDialog() {
    // Reset notification count when user opens notifications
    if (_notificationCount > 0) {
      setState(() {
        _notificationCount = 0;
      });
    }
    
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          backgroundColor: widget.isDarkMode ? Colors.grey[800] : Colors.white,
          title: Row(
            children: [
              Icon(Icons.notifications, color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4)),
              const SizedBox(width: 12),
              Text(
                'Notifications',
                style: TextStyle(
                  color: widget.isDarkMode ? Colors.white : const Color(0xFF00BCD4),
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          content: Container(
            width: MediaQuery.of(context).size.width * 0.8,
            height: MediaQuery.of(context).size.height * 0.6,
            child: Column(
              children: [
                if (_notifications.isEmpty)
                  Expanded(
                    child: Center(
                      child: Text(
                        'No notifications yet.',
                        style: TextStyle(
                          color: widget.isDarkMode ? Colors.white70 : Colors.black54,
                          fontSize: 16,
                        ),
                      ),
                    ),
                  )
                else
                  Expanded(
                    child: ListView.builder(
                      itemCount: _notifications.length,
                      itemBuilder: (context, index) {
                        final notification = _notifications[_notifications.length - 1 - index]; // Show newest first
                        final title = notification['title'] as String?;
                        final message = notification['message'] as String?;
                        final timestamp = notification['timestamp'] as DateTime?;
                        final type = notification['type'] as String? ?? 'info';
                        
                        IconData icon;
                        Color iconColor;
                        
                        switch (type) {
                          case 'success':
                            icon = Icons.check_circle;
                            iconColor = Colors.green;
                            break;
                          case 'warning':
                            icon = Icons.warning;
                            iconColor = Colors.orange;
                            break;
                          case 'error':
                            icon = Icons.error;
                            iconColor = Colors.red;
                            break;
                          default: // info
                            icon = Icons.info;
                            iconColor = const Color(0xFF00BCD4);
                            break;
                        }
                        
                        return Card(
                          color: widget.isDarkMode ? Colors.grey[700] : Colors.white,
                          margin: const EdgeInsets.symmetric(vertical: 4),
                          child: ListTile(
                            leading: Icon(icon, color: iconColor),
                            title: Text(
                              title ?? 'Notification',
                              style: TextStyle(
                                color: widget.isDarkMode ? Colors.white : Colors.black87,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            subtitle: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  message ?? '',
                                  style: TextStyle(
                                    color: widget.isDarkMode ? Colors.grey[300] : Colors.grey[600],
                                    fontSize: 12,
                                  ),
                                ),
                                if (timestamp != null)
                                  Text(
                                    '${timestamp.month.toString().padLeft(2, '0')}/${timestamp.day.toString().padLeft(2, '0')}/${timestamp.year} ${timestamp.hour.toString().padLeft(2, '0')}:${timestamp.minute.toString().padLeft(2, '0')}',
                                    style: TextStyle(
                                      color: widget.isDarkMode ? Colors.grey[400] : Colors.grey[500],
                                      fontSize: 11,
                                    ),
                                  ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                if (_notifications.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 16),
                    child: ElevatedButton(
                      onPressed: () {
                        setState(() {
                          _notifications.clear();
                        });
                        // Show success message
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text('All notifications cleared successfully',
                                style: TextStyle(
                                  color: widget.isDarkMode ? Colors.white : Colors.black87,
                                )),
                            backgroundColor: widget.isDarkMode ? Colors.grey[700] : Colors.grey[300],
                            duration: const Duration(seconds: 2),
                          ),
                        );
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: widget.isDarkMode ? Colors.redAccent : Colors.red,
                        foregroundColor: Colors.white,
                      ),
                      child: const Text('Clear All Notifications'),
                    ),
                  ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text(
                'Close',
                style: TextStyle(color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4), fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ),
          ],
        );
      },
    );
  }

  void _showHistoryDialog() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          backgroundColor: widget.isDarkMode ? Colors.grey[800] : Colors.white,
          title: Row(
            children: [
              Icon(Icons.history, color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4)),
              const SizedBox(width: 12),
              Text(
                'History',
                style: TextStyle(
                  color: widget.isDarkMode ? Colors.white : const Color(0xFF00BCD4),
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          content: Container(
            width: MediaQuery.of(context).size.width * 0.8,
            height: MediaQuery.of(context).size.height * 0.6,
            child: Column(
              children: [
                if (_classificationHistory.isEmpty)
                  Expanded(
                    child: Center(
                      child: Text(
                        'No classification history yet.',
                        style: TextStyle(
                          color: widget.isDarkMode ? Colors.white70 : Colors.black54,
                          fontSize: 16,
                        ),
                      ),
                    ),
                  )
                else
                  Expanded(
                    child: ListView.builder(
                      itemCount: _classificationHistory.length,
                      itemBuilder: (context, index) {
                        final item = _classificationHistory[index];
                        final bird = item['bird'] as String?;
                        final confidence = item['confidence'] as double?;
                        final timestamp = item['timestamp'] as DateTime?;
                        final imagePath = item['imagePath'] as String?;
                        
                        return Card(
                          color: widget.isDarkMode ? Colors.grey[700] : Colors.white,
                          margin: const EdgeInsets.symmetric(vertical: 4),
                          child: ListTile(
                            leading: imagePath != null
                                ? Container(
                                    width: 50,
                                    height: 50,
                                    decoration: BoxDecoration(
                                      borderRadius: BorderRadius.circular(8),
                                      image: DecorationImage(
                                        image: FileImage(File(imagePath)),
                                        fit: BoxFit.cover,
                                      ),
                                    ),
                                  )
                                : Icon(Icons.image, color: widget.isDarkMode ? Colors.white70 : Colors.black54),
                            title: Text(
                              bird ?? 'Unknown',
                              style: TextStyle(
                                color: widget.isDarkMode ? Colors.white : Colors.black87,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            subtitle: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  '${confidence?.toStringAsFixed(1) ?? '0'}% confidence',
                                  style: TextStyle(
                                    color: widget.isDarkMode ? Colors.grey[300] : Colors.grey[600],
                                    fontSize: 12,
                                  ),
                                ),
                                Text(
                                  timestamp != null 
                                      ? '${timestamp.month.toString().padLeft(2, '0')}/${timestamp.day.toString().padLeft(2, '0')}/${timestamp.year} ${timestamp.hour.toString().padLeft(2, '0')}:${timestamp.minute.toString().padLeft(2, '0')}' 
                                      : 'Unknown time',
                                  style: TextStyle(
                                    color: widget.isDarkMode ? Colors.grey[400] : Colors.grey[500],
                                    fontSize: 11,
                                  ),
                                ),
                              ],
                            ),
                            trailing: IconButton(
                              icon: Icon(Icons.delete, color: widget.isDarkMode ? Colors.redAccent : Colors.red),
                              onPressed: () {
                                setState(() {
                                  _classificationHistory.removeAt(index);
                                });
                                // Show success message
                                ScaffoldMessenger.of(context).showSnackBar(
                                  SnackBar(
                                    content: Text('History item deleted successfully',
                                        style: TextStyle(
                                          color: widget.isDarkMode ? Colors.white : Colors.black87,
                                        )),
                                    backgroundColor: widget.isDarkMode ? Colors.grey[700] : Colors.grey[300],
                                    duration: const Duration(seconds: 2),
                                  ),
                                );
                              },
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                if (_classificationHistory.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 16),
                    child: ElevatedButton(
                      onPressed: () {
                        setState(() {
                          _classificationHistory.clear();
                        });
                        // Show success message
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text('All history cleared successfully',
                                style: TextStyle(
                                  color: widget.isDarkMode ? Colors.white : Colors.black87,
                                )),
                            backgroundColor: widget.isDarkMode ? Colors.grey[700] : Colors.grey[300],
                            duration: const Duration(seconds: 2),
                          ),
                        );
                      },
                      style: ElevatedButton.styleFrom(
                        backgroundColor: widget.isDarkMode ? Colors.redAccent : Colors.red,
                        foregroundColor: Colors.white,
                      ),
                      child: const Text('Clear All History'),
                    ),
                  ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text(
                'Close',
                style: TextStyle(color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4), fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ),
          ],
        );
      },
    );
  }

  // Build analytics graph showing all predictions
  Widget _buildAnalyticsGraph() {
    if (_allProbabilities == null || _labels == null) return const SizedBox.shrink();

    // Create a list of (index, probability) pairs and sort by probability
    List<MapEntry<int, double>> indexedProbabilities = [];
    for (int i = 0; i < _allProbabilities!.length; i++) {
      indexedProbabilities.add(MapEntry(i, _allProbabilities![i]));
    }
    
    // Sort by probability in descending order
    indexedProbabilities.sort((a, b) => b.value.compareTo(a.value));

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: widget.isDarkMode ? const Color(0xFF424242) : Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: widget.isDarkMode ? Colors.black.withOpacity(0.3) : Colors.black.withOpacity(0.05),
            spreadRadius: 0,
            blurRadius: 20,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Color(0xFF00BCD4), Color(0xFF00ACC1)],
                  ),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: const Icon(Icons.analytics_outlined, color: Colors.white, size: 20),
              ),
              const SizedBox(width: 12),
              Text(
                'Prediction Analytics',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w600,
                  color: widget.isDarkMode ? Colors.white : Color(0xFF1A1A1A),
                  letterSpacing: -0.5,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            'Confidence distribution across all classes',
            style: TextStyle(
              fontSize: 13,
              color: widget.isDarkMode ? Colors.grey[300] : Colors.grey[600],
              fontWeight: FontWeight.w400,
            ),
          ),
          const SizedBox(height: 24),
          ...indexedProbabilities.map((entry) {
            int index = entry.key;
            double probability = entry.value * 100;
            String birdName = _extractBirdName(_labels![index]);
            bool isTopPrediction = index == _predictedIndex;
            
            return Padding(
              padding: const EdgeInsets.only(bottom: 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Expanded(
                        child: Text(
                          birdName,
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: isTopPrediction ? FontWeight.w600 : FontWeight.w400,
                            color: isTopPrediction ? (widget.isDarkMode ? Colors.white : const Color(0xFF1A1A1A)) : (widget.isDarkMode ? Colors.grey[300] : const Color(0xFF666666)),
                            letterSpacing: -0.2,
                          ),
                        ),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                        decoration: BoxDecoration(
                          color: isTopPrediction ? const Color(0xFF00BCD4).withOpacity(0.1) : Colors.grey[100],
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Text(
                          '${probability.toStringAsFixed(1)}%',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: isTopPrediction ? const Color(0xFF00BCD4) : Colors.grey[700],
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(6),
                    child: Stack(
                      children: [
                        Container(
                          height: 6,
                          decoration: BoxDecoration(
                            color: Colors.grey[100],
                          ),
                        ),
                        AnimatedContainer(
                          duration: const Duration(milliseconds: 600),
                          curve: Curves.easeOutCubic,
                          height: 6,
                          width: (MediaQuery.of(context).size.width - 96) * (probability / 100),
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: isTopPrediction 
                                  ? [const Color(0xFF00BCD4), const Color(0xFF00ACC1)]
                                  : [Colors.grey[300]!, Colors.grey[400]!],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            );
          }).toList(),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return Scaffold(
        body: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [Color(0xFF00BCD4), Color(0xFF00ACC1)],
            ),
          ),
          child: const Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                CircularProgressIndicator(
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                  strokeWidth: 6,
                ),
                SizedBox(height: 20),
                Text(
                  'Loading Bird Classifier...',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    }

    // Handle model loading error
    if (_modelLoadError) {
      return Scaffold(
        body: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [Color(0xFF00BCD4), Color(0xFF00ACC1)],
            ),
          ),
          child: Center(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(
                    Icons.error_outline,
                    color: Colors.white,
                    size: 60,
                  ),
                  const SizedBox(height: 20),
                  const Text(
                    'Failed to Load Model',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    _modelErrorMessage,
                    style: const TextStyle(
                      color: Colors.white70,
                      fontSize: 16,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 30),
                  ElevatedButton(
                    onPressed: () {
                      setState(() {
                        _loading = true;
                        _modelLoadError = false;
                        _modelErrorMessage = '';
                      });
                      _loadModel();
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: const Color(0xFF00BCD4),
                      padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(30),
                      ),
                    ),
                    child: const Text(
                      'Retry',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    }

    print("Building Scaffold with drawer");
    return Scaffold(
      backgroundColor: widget.isDarkMode ? Colors.grey[900] : const Color(0xFFF0F8FF),
      appBar: AppBar(
        centerTitle: true,
        flexibleSpace: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [
                Color(0xFF00BCD4),
                Color(0xFF00ACC1),
              ],
            ),
          ),
        ),
        leading: Builder(
          builder: (context) => IconButton(
            icon: Icon(Icons.menu, color: widget.isDarkMode ? Colors.white : Colors.white, size: 28),
            tooltip: 'Menu',
            onPressed: () {
              Scaffold.of(context).openDrawer();
            },
          ),
        ),
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: (widget.isDarkMode ? Colors.white : Colors.white).withOpacity(0.2),
                borderRadius: BorderRadius.circular(10),
              ),
              child: _buildBirdLogo(size: 24, color: widget.isDarkMode ? Colors.white : Colors.white),
            ),
            const SizedBox(width: 12),
            Text(
              'Bird Classifier',
              style: TextStyle(
                color: widget.isDarkMode ? Colors.white : Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        elevation: 0,
        actions: [
          Stack(
            children: [
              IconButton(
                icon: Icon(Icons.notifications_outlined, color: widget.isDarkMode ? Colors.white : Colors.white, size: 28),
                tooltip: 'Notifications',
                onPressed: _showNotificationsDialog,
              ),
              if (_notificationCount > 0)
                Positioned(
                  right: 8,
                  top: 8,
                  child: Container(
                    padding: const EdgeInsets.all(4),
                    decoration: BoxDecoration(
                      color: Colors.red,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    constraints: const BoxConstraints(
                      minWidth: 16,
                      minHeight: 16,
                    ),
                    child: Text(
                      _notificationCount > 9 ? '9+' : '$_notificationCount',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),
            ],
          ),
          IconButton(
            icon: Icon(Icons.settings_outlined, color: widget.isDarkMode ? Colors.white : Colors.white, size: 28),
            tooltip: 'Settings',
            onPressed: _showSettingsDialog,
          ),
          IconButton(
            icon: Icon(Icons.info_outline, color: widget.isDarkMode ? Colors.white : Colors.white, size: 28),
            tooltip: 'About',
            onPressed: () {
              showDialog(
                context: context,
                builder: (BuildContext context) {
                  return AlertDialog(
                    backgroundColor: widget.isDarkMode ? Colors.grey[800] : Colors.white,
                    title: Text(
                      'About This App',
                      style: TextStyle(color: widget.isDarkMode ? Colors.white : const Color(0xFF00BCD4)),
                    ),
                    content: Text(
                      'Bird Classifier uses AI to identify exotic birds from photos. Simply upload a photo or use our example images to see the magic happen!\n\nVersion: 1.0.0\nPowered by TensorFlow Lite',
                      style: TextStyle(
                        color: widget.isDarkMode ? Colors.white : Colors.black87,
                      ),
                    ),
                    actions: [
                      TextButton(
                        onPressed: () => Navigator.pop(context),
                        child: Text(
                          'Got it!',
                          style: TextStyle(color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4)),
                        ),
                      ),
                    ],
                  );
                },
              );
            },
          ),
        ],
      ),
      drawer: Drawer(
        child: Container(
          color: widget.isDarkMode ? Colors.grey[850] : const Color(0xFFF0F8FF),
          child: Column(
            children: [
              Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      Color(0xFF00BCD4),
                      Color(0xFF0097A7),
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),
                child: SafeArea(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        _buildBirdLogo(size: 36, color: Colors.white),
                        const SizedBox(height: 10),
                        const Text(
                          'Bird Classifier',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          'AI-Powered Recognition',
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.white.withOpacity(0.9),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              Expanded(
                child: ListView(
                  padding: const EdgeInsets.symmetric(vertical: 8),
                  shrinkWrap: true,
                  children: [
                    ListTile(
                      leading: Icon(Icons.photo_library, color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4)),
                      title: Text(
                        'Example Images',
                        style: TextStyle(
                          color: widget.isDarkMode ? Colors.white : Colors.black87,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      subtitle: Text(
                        'Browse sample birds',
                        style: TextStyle(
                          color: widget.isDarkMode ? Colors.grey[400] : Colors.grey,
                        ),
                      ),
                      onTap: () {
                        Navigator.pop(context);
                        _showExamplesBottomSheet();
                      },
                    ),
                    ListTile(
                      leading: Icon(Icons.settings, color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4)),
                      title: Text(
                        'Settings',
                        style: TextStyle(
                          color: widget.isDarkMode ? Colors.white : Colors.black87,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      subtitle: Text(
                        'Configure app preferences',
                        style: TextStyle(
                          color: widget.isDarkMode ? Colors.grey[400] : Colors.grey,
                        ),
                      ),
                      onTap: () {
                        Navigator.pop(context);
                        _showSettingsDialog();
                      },
                    ),
                    ListTile(
                      leading: Icon(Icons.notifications, color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4)),
                      title: Text(
                        'Notifications',
                        style: TextStyle(
                          color: widget.isDarkMode ? Colors.white : Colors.black87,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      subtitle: Text(
                        'View notification history',
                        style: TextStyle(
                          color: widget.isDarkMode ? Colors.grey[400] : Colors.grey,
                        ),
                      ),
                      onTap: () {
                        Navigator.pop(context);
                        _showNotificationsDialog();
                      },
                    ),
                    ListTile(
                      leading: Icon(Icons.history, color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4)),
                      title: Text(
                        'History',
                        style: TextStyle(
                          color: widget.isDarkMode ? Colors.white : Colors.black87,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      subtitle: Text(
                        'View classification history',
                        style: TextStyle(
                          color: widget.isDarkMode ? Colors.grey[400] : Colors.grey,
                        ),
                      ),
                      onTap: () {
                        Navigator.pop(context);
                        _showHistoryDialog();
                      },
                    ),
                    ListTile(
                      leading: Icon(Icons.info_outline, color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4)),
                      title: Text(
                        'About',
                        style: TextStyle(
                          color: widget.isDarkMode ? Colors.white : Colors.black87,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      subtitle: Text(
                        'Learn about this app',
                        style: TextStyle(
                          color: widget.isDarkMode ? Colors.grey[400] : Colors.grey,
                        ),
                      ),
                      onTap: () {
                        Navigator.pop(context);
                        showDialog(
                          context: context,
                          builder: (BuildContext context) {
                            return AlertDialog(
                              backgroundColor: widget.isDarkMode ? Colors.grey[800] : Colors.white,
                              title: Row(
                                children: [
                                  Icon(Icons.info_outline, color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4)),
                                  const SizedBox(width: 12),
                                  Text(
                                    'About This App',
                                    style: TextStyle(
                                      color: widget.isDarkMode ? Colors.white : const Color(0xFF00BCD4),
                                      fontSize: 20,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ],
                              ),
                              content: Text(
                                'Bird Classifier uses AI to identify exotic birds from photos. Simply upload a photo or use our example images to see the magic happen!\n\nVersion: 1.0.0\nPowered by TensorFlow Lite',
                                style: TextStyle(
                                  color: widget.isDarkMode ? Colors.white : Colors.black87,
                                ),
                              ),
                              actions: [
                                TextButton(
                                  onPressed: () => Navigator.pop(context),
                                  child: Text(
                                    'Got it!',
                                    style: TextStyle(color: widget.isDarkMode ? const Color(0xFF00BCD4) : const Color(0xFF00BCD4), fontSize: 16, fontWeight: FontWeight.bold),
                                  ),
                                ),
                              ],
                            );
                          },
                        );
                      },
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              // Welcome card
              Container(
                decoration: BoxDecoration(
                  color: widget.isDarkMode ? const Color(0xFF424242) : Colors.white,
                  borderRadius: BorderRadius.circular(24),
                  boxShadow: [
                    BoxShadow(
                      color: widget.isDarkMode ? Colors.black.withOpacity(0.3) : Colors.black.withOpacity(0.1),
                      spreadRadius: 0,
                      blurRadius: 20,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Padding(
                  padding: const EdgeInsets.all(32),
                  child: Column(
                    children: [
                      // App Logo
                      Container(
                        padding: const EdgeInsets.all(20),
                        decoration: BoxDecoration(
                          gradient: const LinearGradient(
                            colors: [Color(0xFF00BCD4), Color(0xFF00ACC1)],
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                          ),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: _buildBirdLogo(size: 48, color: Colors.white),
                      ),
                      const SizedBox(height: 20),
                      Text(
                        'Identify Exotic Birds',
                        style: TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.w600,
                          color: widget.isDarkMode ? Colors.white : const Color(0xFF1A1A1A),
                          letterSpacing: -0.5,
                        ),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Upload a photo or try our examples',
                        style: TextStyle(
                          fontSize: 15,
                          color: widget.isDarkMode ? Colors.grey[300] : Colors.grey[600],
                          fontWeight: FontWeight.w400,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 25),
              
              // Upload button
              Container(
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Color(0xFF00BCD4), Color(0xFF00ACC1)],
                  ),
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: [
                    BoxShadow(
                      color: const Color(0xFF00BCD4).withOpacity(0.3),
                      spreadRadius: 0,
                      blurRadius: 12,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: ElevatedButton.icon(
                  onPressed: _showPickerDialog,
                  icon: const Icon(Icons.image_outlined, color: Colors.white, size: 22),
                  label: const Padding(
                    padding: EdgeInsets.symmetric(vertical: 16),
                    child: Text(
                      'Select Bird Photo',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: Colors.white,
                        letterSpacing: -0.3,
                      ),
                    ),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.transparent,
                    shadowColor: Colors.transparent,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 30),
              
              // Display selected image if available
              if (_imageFile != null) ...[
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: widget.isDarkMode ? const Color(0xFF424242) : Colors.white,
                    borderRadius: BorderRadius.circular(24),
                    boxShadow: [
                      BoxShadow(
                        color: widget.isDarkMode ? Colors.black.withOpacity(0.3) : Colors.black.withOpacity(0.1),
                        spreadRadius: 0,
                        blurRadius: 20,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(16),
                    child: Image.file(
                      _imageFile!,
                      fit: BoxFit.contain,
                      errorBuilder: (context, error, stackTrace) {
                        print("Error displaying image: $error");
                        return Container(
                          height: 300,
                          color: Colors.grey[100],
                          child: const Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(Icons.broken_image, size: 100, color: Colors.grey),
                              SizedBox(height: 10),
                              Text(
                                'Failed to display image',
                                style: TextStyle(color: Colors.grey),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
                  ),
                ),
                const SizedBox(height: 25),
              ],
              
              // Display prediction result
              if (_prediction != null) ...[
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(32),
                  decoration: BoxDecoration(
                    color: widget.isDarkMode ? const Color(0xFF424242) : Colors.white,
                    borderRadius: BorderRadius.circular(24),
                    boxShadow: [
                      BoxShadow(
                        color: widget.isDarkMode ? Colors.black.withOpacity(0.3) : Colors.black.withOpacity(0.1),
                        spreadRadius: 0,
                        blurRadius: 20,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: Column(
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                        decoration: BoxDecoration(
                          color: const Color(0xFF00BCD4).withOpacity(0.1),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: const Text(
                          'Identified Bird',
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: Color(0xFF00BCD4),
                            letterSpacing: -0.2,
                          ),
                        ),
                      ),
                      const SizedBox(height: 16),
                      Text(
                        _prediction!,
                        style: TextStyle(
                          fontSize: 32,
                          fontWeight: FontWeight.w700,
                          color: widget.isDarkMode ? Colors.white : const Color(0xFF1A1A1A),
                          letterSpacing: -1,
                        ),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 24),
                      _buildConfidenceIndicator(_confidence!),
                    ],
                  ),
                ),
                const SizedBox(height: 25),
                
                // Top predictions with descriptions
                _buildTopPredictions(),
                const SizedBox(height: 25),
                
                // Analytics graph
                if (_showAnalytics) ...[
                  _buildAnalyticsGraph(),
                  const SizedBox(height: 25),
                ],
              ],
              
              // Show example images section
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: widget.isDarkMode ? const Color(0xFF424242) : Colors.white,
                  borderRadius: BorderRadius.circular(24),
                  boxShadow: [
                    BoxShadow(
                      color: widget.isDarkMode ? Colors.black.withOpacity(0.3) : Colors.black.withOpacity(0.05),
                      spreadRadius: 0,
                      blurRadius: 20,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            gradient: const LinearGradient(
                              colors: [Color(0xFF00BCD4), Color(0xFF00ACC1)],
                            ),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: _buildBirdLogo(size: 16, color: Colors.white),
                        ),
                        const SizedBox(width: 12),
                        Text(
                          'Quick Examples',
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.w600,
                            color: widget.isDarkMode ? Colors.white : const Color(0xFF1A1A1A),
                            letterSpacing: -0.5,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Try these sample bird images',
                      style: TextStyle(
                        fontSize: 13,
                        color: widget.isDarkMode ? Colors.grey[300] : Colors.grey[600],
                      ),
                    ),
                    const SizedBox(height: 20),
                    GridView.count(
                      shrinkWrap: true,
                      physics: const NeverScrollableScrollPhysics(),
                      crossAxisCount: 3,
                      crossAxisSpacing: 12,
                      mainAxisSpacing: 12,
                      children: List.generate(6, (index) {
                        String imageName = _getImageName(index);
                        String birdName = _labels != null && _labels!.length > index 
                            ? _labels![index].split(' ')[1] 
                            : 'Bird ${index + 1}';
                        print("Quick example $index: $imageName, Bird: $birdName");
                        return GestureDetector(
                          onTap: () {
                            // Add error handling for quick examples
                            try {
                              print("Tapped quick example $index: $imageName");
                              _loadExampleImage(imageName);
                            } catch (e) {
                              print("Error loading quick example $index: $e");
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(content: Text('Error loading example: ${e.toString()}')),
                              );
                            }
                          },
                          child: Hero(
                            tag: 'example_$index',
                            child: Card(
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(12),
                                side: BorderSide(color: Colors.grey[200]!, width: 1),
                              ),
                              child: Column(
                                children: [
                                  Expanded(
                                    child: ClipRRect(
                                      borderRadius: const BorderRadius.vertical(top: Radius.circular(12)),
                                      child: Image.asset(
                                        'assets/examples/$imageName',
                                        fit: BoxFit.cover,
                                        width: double.infinity,
                                        errorBuilder: (context, error, stackTrace) {
                                          print("Error loading quick example image $index ($imageName): $error");
                                          return Container(
                                            color: Colors.grey[100],
                                            child: const Icon(Icons.error_outline, size: 24, color: Colors.grey),
                                          );
                                        },
                                      ),
                                    ),
                                  ),
                                  Container(
                                    padding: const EdgeInsets.symmetric(vertical: 8),
                                    child: Text(
                                      birdName,
                                      style: TextStyle(
                                        color: widget.isDarkMode ? Colors.white : const Color(0xFF1A1A1A),
                                        fontSize: 11,
                                        fontWeight: FontWeight.w600,
                                        letterSpacing: -0.2,
                                      ),
                                      textAlign: TextAlign.center,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        );
                      }),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 10),
            ],
          ),
        ),
      ),
    );
  }
}